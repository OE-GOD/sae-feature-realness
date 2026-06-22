"""Harder-shift harness: reuses the VALIDATED context H from idea_harness (same train/rt,
same reliable features, same full probe) but tests on genuinely different registers —
tweet / poem / financial — instead of the review OOD domains. Tests the frontier caveat:
does two-view disagreement still win when the shift is semantic/register, not vocabulary?
Same idea contract: make_signal(H, pool) -> score(Ztest) -> trust array (higher = keep)."""
import torch, numpy as np
from idea_harness import H, _auroc, _augrc

_hs = torch.load("/Users/oe/rebuild/hard_shift_sentiment.pt", weights_only=False)
POOLINGS = ["mean", "max"]
HARD = {k: v for k, v in _hs.items()}  # tweet, poem, financial

def run_hard(make_signal, name):
    print(f"\n=== {name} (HARDER SHIFT) ===")
    print(f"{'pool':6}{'domain':10}{'base':>6}{'AUROC':>7}{'AUGRC':>7}"
          f"{'cov':>6}{'CAND':>7}{'conf':>7}{'rand':>7}{'win?':>6}")
    wins = 0; total = 0; rows = []
    for pool in POOLINGS:
        h = H(pool)
        try:
            score = make_signal(h, pool)
        except Exception as e:
            print(f"{pool:6}make_signal FAILED: {e}"); continue
        for dom, (P, yt) in HARD.items():
            Z = h.std(P[pool].float().numpy()); y = yt.numpy()
            pf = h.full.predict(Z); correct = (pf == y).astype(int); base = correct.mean()
            trust = np.asarray(score(Z), dtype=float)
            au = _auroc(trust, correct); ag = _augrc(trust, correct)
            uniq = np.unique(trust)
            if len(uniq) <= 2:
                keep = trust >= uniq.max(); cov = keep.mean()
            else:
                cov = 0.5; thr = np.quantile(trust, 1 - cov); keep = trust >= thr
            cand_acc = correct[keep].mean() if keep.sum() else base
            conf = np.abs(h.full.decision_function(Z))
            cthr = np.quantile(conf, 1 - cov); conf_acc = correct[conf >= cthr].mean() if (conf >= cthr).sum() else base
            rnds = []
            for s in range(5):
                rr = np.random.RandomState(s).choice(h.alive, len(h.reliable), replace=False)
                rp = h.probe(cols=rr); ra = (pf == rp.predict(Z[:, rr]))
                rnds.append(correct[ra].mean() if ra.sum() else base)
            rand_acc = np.mean(rnds)
            win = cand_acc > conf_acc + 1e-9 and cand_acc > rand_acc + 1e-9
            wins += win; total += 1
            rows.append((pool, dom, au, ag, cov, cand_acc, conf_acc, rand_acc, win))
            print(f"{pool:6}{dom:10}{base:>6.3f}{au:>7.3f}{ag:>7.3f}{cov:>6.2f}"
                  f"{cand_acc:>7.3f}{conf_acc:>7.3f}{rand_acc:>7.3f}{'  Y' if win else '  .':>6}")
    print(f"--- {name}: beats BOTH confidence & random on {wins}/{total} HARDER-shift conditions ---")
    return {"name": name, "wins": wins, "total": total, "rows": rows}
