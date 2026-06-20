"""
feature_realness.py
===================
Score each SAE feature by how "real" it is, using a MAJORITY vote across
several *independent* and *cheap* realness axes. No retraining; everything
is computed from the existing artifacts (sae.pt, sae2.pt, thoughts.pt) plus
one short forward pass of pythia-160m to recover token ids.

HONESTY / EPISTEMIC LIMITS (read this):
  * There is NO oracle for whether an individual feature is "real". We cannot
    certify any single feature. The best we can do is score features on several
    independent axes (which genuinely *disagree* with each other) and flag the
    ones that pass a MAJORITY as "mostly_real". "mostly_real" means
    "passes a strict majority of independent cheap proxies", NOT "proven real".
  * STABILITY (cross-seed decoder replication) is treated as the strongest
    single proxy: a direction that two independently-seeded SAEs both learn is
    unlikely to be a pure fitting artifact. We therefore (a) include it as a
    vote and (b) require it as part of the majority being meaningful, but we do
    NOT make it a hard single-axis gate, because stability is known to be
    blind to causal usage and can reject real features that "split" across
    seeds. Majority voting hedges against any one axis being wrong.
  * FREQUENCY is NOT a positive realness vote. High firing rate can mean a
    dense/polysemantic/punctuation feature; ultra-low means under-sampled and
    un-estimable. So frequency is used only as an ALIVE GATE (drop dead /
    barely-firing features whose interpretability is undefined) and reported
    as a covariate to check the flagged set isn't just "the frequent ones".
  * CAUSAL ablation is intentionally OMITTED from the per-feature score: it
    costs one forward pass per feature. The cons of every cheap design stand:
    a flagged feature may replicate + look clean yet be causally inert. A
    sampled ablation spot-check on the shortlist is the recommended follow-up.

THE THREE VOTING AXES (each independent, cheap, computed for every alive feature):
  1. STABILITY      = cross-seed best decoder-column cosine (sae vs sae2).
                      Pure linear algebra on the two checkpoints. high=replicates.
  2. INTERPRETABILITY = top-3 token concentration of a feature's firings.
                      Needs token ids -> one short text pass. high=monosemantic.
  3. LOGIT-COHERENCE = decoder direction through unembedding (W_dec[:,f] @ W_U),
                      top-10 abs-mass fraction. Needs only W_U. high=clean output.

THRESHOLDS (ALL PROVISIONAL -- documented inline):
  Data-driven per-axis cutoff = the MEDIAN of that axis among ALIVE features.
  Rationale: no magic absolute constants leak in, and the cutoff adapts to
  the (here deliberately tiny/under-trained) SAE. CAVEAT: a median split labels
  ~half of alive features as passing *each* axis almost by construction, so the
  absolute positive rate is inflated and is illustrative, not authoritative.
  For a real SAE, replace the median with a held-out high percentile (e.g. the
  66th) or an absolute bar and report it. The MEDIAN_OR_PERCENTILE knob below
  makes this switch a one-liner.

REALNESS RULE:
  votes = pass_stability + pass_interp + pass_coherence   (0..3)
  mostly_real := alive AND votes >= 2   (strict majority of the 3 axes)
"""

import torch
import torch.nn as nn
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
# Config / provisional knobs                                                   #
# --------------------------------------------------------------------------- #
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
SAE_PATH        = "/Users/oe/rebuild/sae.pt"
SAE2_PATH       = "/Users/oe/rebuild/sae2.pt"
THOUGHTS_PATH   = "/Users/oe/rebuild/thoughts.pt"

N_DOCS          = 60      # pile docs to re-run the model on (for token ids)
DOC_CHARS       = 700     # chars per doc fed to the model
ALIVE_MIN_FIRES = 10      # PROVISIONAL: a feature must fire on >=10 tokens in the
                          # text pass to be scorable (interp is undefined below this).
TOPK_TOKENS     = 3       # interpretability = fraction of firings on top-3 token ids
TOPK_LOGITS     = 10      # logit-coherence = top-10 abs-mass fraction
MAJORITY        = 2       # votes needed (>=2 of 3) to flag mostly_real

# Threshold rule per axis. "median" = median over alive (data-driven, default).
# Set to a float in (0,1) to use that PERCENTILE instead (e.g. 0.66 = top third).
# PROVISIONAL either way -- documented above.
MEDIAN_OR_PERCENTILE = "median"


# --------------------------------------------------------------------------- #
# Exact TinySAE class (must match the checkpoints)                             #
# --------------------------------------------------------------------------- #
class TinySAE(nn.Module):
    def __init__(s, d=768, n=2048, k=32):
        super().__init__()
        s.k = k
        s.W_enc = nn.Linear(d, n)
        s.W_dec = nn.Linear(n, d)

    def forward(s, x):
        sc = s.W_enc(x)
        tk = torch.topk(sc, s.k, -1)
        sp = torch.zeros_like(sc)
        sp.scatter_(-1, tk.indices, tk.values)
        return s.W_dec(sp), sp


def load_sae(path):
    sae = TinySAE().to(DEVICE)
    sae.load_state_dict(torch.load(path, map_location=DEVICE))
    sae.eval()
    return sae


def axis_threshold(values_alive):
    """PROVISIONAL per-axis cutoff over the ALIVE-feature distribution."""
    if MEDIAN_OR_PERCENTILE == "median":
        return values_alive.median().item()
    q = float(MEDIAN_OR_PERCENTILE)
    return torch.quantile(values_alive, q).item()


def main():
    torch.set_grad_enabled(False)

    # ----------------------------------------------------------------- #
    # Load artifacts                                                     #
    # ----------------------------------------------------------------- #
    print("[load] checkpoints + activations ...")
    sae  = load_sae(SAE_PATH)
    sae2 = load_sae(SAE2_PATH)
    thoughts = torch.load(THOUGHTS_PATH, map_location=DEVICE).float()  # [69529,768]

    Wd  = sae.W_dec.weight.detach()    # [768, 2048]  column f = feature f decoder dir
    Wd2 = sae2.W_dec.weight.detach()   # [768, 2048]
    N_FEATURES = Wd.shape[1]
    N_TOK = thoughts.shape[0]

    # ----------------------------------------------------------------- #
    # AXIS 1 -- STABILITY (cheap, weights only; the strongest single proxy)
    #   best cosine of each sae decoder column to ANY sae2 decoder column.
    # ----------------------------------------------------------------- #
    print("[axis] stability (cross-seed decoder cosine) ...")
    W1n = Wd  / (Wd.norm(dim=0,  keepdim=True) + 1e-9)
    W2n = Wd2 / (Wd2.norm(dim=0, keepdim=True) + 1e-9)
    stability = (W1n.t() @ W2n).max(dim=1).values   # [2048] in [-1,1]

    # ----------------------------------------------------------------- #
    # AXIS 3 -- LOGIT-COHERENCE (cheap, needs only model.W_U)
    #   top-10 abs-mass fraction of decoder direction through unembedding.
    # ----------------------------------------------------------------- #
    print("[load] pythia-160m (for W_U and token ids) ...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("pythia-160m").to(DEVICE)
    model.eval()
    HOOK = "blocks.6.hook_resid_post"

    print("[axis] logit-coherence (decoder -> unembedding top-10 mass) ...")
    WU = model.W_U.detach()             # [768, vocab]
    LE = (Wd.t() @ WU).abs()           # [2048, vocab] direct-logit effect magnitude
    coherence = LE.topk(TOPK_LOGITS, dim=1).values.sum(1) / (LE.sum(1) + 1e-9)  # [2048]
    del LE

    # ----------------------------------------------------------------- #
    # AXES 2 + FREQUENCY -- one short text pass to recover token ids
    #   (thoughts.pt has no ids, so we re-run the model on ~60 pile docs).
    #   freq: fraction of tokens a feature fires on (covariate / alive gate)
    #   interp: top-3 token-id concentration of a feature's firings
    # ----------------------------------------------------------------- #
    print(f"[axis] interpretability + frequency ({N_DOCS} pile docs) ...")
    from datasets import load_dataset
    data = load_dataset("NeelNanda/pile-10k", split="train")

    fires      = torch.zeros(N_FEATURES, device=DEVICE)   # total firings per feature
    tok_counts = defaultdict(Counter)                     # feature -> Counter(token_id)
    total_tokens = 0

    for i in range(N_DOCS):
        text = data[i]["text"][:DOC_CHARS]
        if not text.strip():
            continue
        ids = model.to_tokens(text)                       # [1, T]
        _, cache = model.run_with_cache(ids, names_filter=HOOK)
        resid = cache[HOOK][0]                             # [T, 768]
        _, sp = sae(resid)                                # [T, 2048]
        total_tokens += ids.shape[1]

        rows, cols = torch.where(sp > 0)                  # firing positions
        token_ids = ids[0, rows]
        for f, tid in zip(cols.tolist(), token_ids.tolist()):
            tok_counts[f][tid] += 1
        fires += (sp > 0).sum(0).float()

    # interpretability = sum(top-3 token counts) / total firings   (per feature)
    interp = torch.zeros(N_FEATURES, device=DEVICE)
    top_tokens = {}                                       # feature -> [token strings]
    for f, ctr in tok_counts.items():
        tot = sum(ctr.values())
        if tot == 0:
            continue
        top3 = ctr.most_common(TOPK_TOKENS)
        interp[f] = sum(c for _, c in top3) / tot
        top_tokens[f] = [model.tokenizer.decode([tid]) for tid, _ in top3]

    freq = fires / max(total_tokens, 1)                  # covariate, NOT a vote

    # ----------------------------------------------------------------- #
    # ALIVE GATE (frequency used only here -- not a positive vote)        #
    # ----------------------------------------------------------------- #
    alive = fires >= ALIVE_MIN_FIRES                     # bool [2048]
    n_alive = int(alive.sum().item())
    if n_alive == 0:
        raise RuntimeError("No alive features -- increase N_DOCS / lower ALIVE_MIN_FIRES.")

    # ----------------------------------------------------------------- #
    # PROVISIONAL per-axis thresholds (over alive features only)          #
    # ----------------------------------------------------------------- #
    t_stab   = axis_threshold(stability[alive])
    t_interp = axis_threshold(interp[alive])
    t_coh    = axis_threshold(coherence[alive])

    pass_S = (stability >= t_stab) & alive
    pass_I = (interp    >= t_interp) & alive
    pass_C = (coherence >= t_coh)    & alive

    votes = pass_S.int() + pass_I.int() + pass_C.int()   # [2048] in 0..3
    mostly_real = alive & (votes >= MAJORITY)

    # ----------------------------------------------------------------- #
    # Independence sanity: weak pairwise corr => votes are non-redundant  #
    # ----------------------------------------------------------------- #
    def corr(a, b):
        a = a[alive]; b = b[alive]
        a = a - a.mean(); b = b - b.mean()
        d = (a.norm() * b.norm()) + 1e-9
        return (a @ b / d).item()

    # ----------------------------------------------------------------- #
    # Report                                                             #
    # ----------------------------------------------------------------- #
    n_real = int(mostly_real.sum().item())
    print("\n" + "=" * 70)
    print("FEATURE REALNESS -- MAJORITY-VOTE BATTERY (PROVISIONAL)")
    print("=" * 70)
    print(f"features total                 : {N_FEATURES}")
    print(f"alive (fires>={ALIVE_MIN_FIRES})            : {n_alive}"
          f"  ({n_alive/N_FEATURES:.1%})  [{N_FEATURES-n_alive} unscorable, dropped]")
    print(f"text pass                      : {N_DOCS} docs, {total_tokens} tokens")
    print(f"threshold rule                 : {MEDIAN_OR_PERCENTILE} over alive (PROVISIONAL)")
    print(f"  stability  >= {t_stab:.4f}")
    print(f"  interp     >= {t_interp:.4f}")
    print(f"  coherence  >= {t_coh:.4f}")
    print(f"per-axis pass (of alive)       : "
          f"S={int(pass_S.sum())}  I={int(pass_I.sum())}  C={int(pass_C.sum())}")
    print(f"axis independence (corr,alive) : "
          f"S-I={corr(stability,interp):+.2f}  "
          f"S-C={corr(stability,coherence):+.2f}  "
          f"I-C={corr(interp,coherence):+.2f}  (weak => non-redundant votes)")
    print(f"freq confound check            : "
          f"corr(freq,stability)={corr(freq,stability):+.2f} "
          f"(near 0 => flags not just 'frequent ones')")
    print("-" * 70)
    print(f">>> FRACTION MOSTLY-REAL (of alive): {n_real}/{n_alive} = "
          f"{n_real/n_alive:.1%}   (of all {N_FEATURES}: {n_real/N_FEATURES:.1%})")
    print("    'mostly_real' = passes >=2 of 3 independent axes. NOT certified real.")
    print("=" * 70)

    # ----------------------------------------------------------------- #
    # Examples                                                           #
    # ----------------------------------------------------------------- #
    def describe(f):
        toks = top_tokens.get(f, ["<no firings>"])
        toks = [repr(t) for t in toks]
        return (f"  feat {f:4d} | votes={int(votes[f])} "
                f"S={'Y' if pass_S[f] else '.'}"
                f"I={'Y' if pass_I[f] else '.'}"
                f"C={'Y' if pass_C[f] else '.'} "
                f"| stab={stability[f]:.3f} interp={interp[f]:.3f} "
                f"coh={coherence[f]:.4f} freq={freq[f]:.4f}\n"
                f"           top tokens: {', '.join(toks)}")

    real_ids = mostly_real.nonzero(as_tuple=True)[0].tolist()
    notreal_ids = (alive & ~mostly_real).nonzero(as_tuple=True)[0].tolist()
    # sort real by vote-count then stability (most-confident first)
    real_ids.sort(key=lambda f: (int(votes[f]), float(stability[f])), reverse=True)
    notreal_ids.sort(key=lambda f: (int(votes[f]), float(stability[f])))

    print("\n3 EXAMPLE MOSTLY-REAL FEATURES (most confident first):")
    for f in real_ids[:3]:
        print(describe(f))

    print("\n3 EXAMPLE NOT-REAL (alive but <2 votes):")
    for f in notreal_ids[:3]:
        print(describe(f))

    print("\nNOTE: results are from a deliberately tiny/under-trained TinySAE; "
          "absolute numbers are illustrative. Recommended follow-up: sampled "
          "CAUSAL ablation on the mostly_real shortlist to catch stable+clean "
          "but causally-inert features (the failure mode the cheap axes can't see).")


if __name__ == "__main__":
    main()
