"""
Adversarial verification of the WINNING Gemma Scope detector recipe
(l1|nf400|mlp) used by gemma_best_detectors.py.

For a given seed it reproduces the exact recipe (L1 logreg selection of top-400
of 16384 SAE features -> standardize -> MLP(32) -> F1-optimal threshold on train)
and reports, per concept and per OOD split (TinyStories, wikitext):
  base rate, precision, recall, F1, predicted-positive rate, and the
  always-positive baseline F1 (= F1 you'd get by predicting 1 everywhere).

Flags:
  DEGENERATE  if predicted-positive rate >= 0.97 (predictor is basically
              always-positive) OR detector F1 - always-positive F1 < 0.02
  NO_OOD_POS  if the split has < 3 positives (recipe drops it; not a real test)
"""
import sys, torch, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(SEED); np.random.seed(SEED)

DATA = "/Users/oe/rebuild/gemma_detector_dataset.pt"
NF = 400

d = torch.load(DATA, map_location="cpu")
CONCEPTS = d["keys"]
train_F = d["train_F"].float().numpy()
splits = {
    "indist": (d["test_F"].float().numpy(), d["test_L"]),
    "ts":     (d["ts_F"].float().numpy(),   d["ts_L"]),
    "wk":     (d["wk_F"].float().numpy(),   d["wk_L"]),
}


def prf(y, pred):
    y = y.astype(int); pred = pred.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def best_thresh(y, p):
    cand = np.unique(p)
    ths = (np.concatenate([[-1e9], (cand[:-1] + cand[1:]) / 2, [1e9]])
           if len(cand) > 1 else np.array([0.5]))
    bf, bt = -1.0, 0.5
    for t in ths:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return bt


rows = []
ood_f1_for_mean = []   # mirrors gemma_best_detectors.py averaging (all ts+wk that survive)
for C in CONCEPTS:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue

    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                            max_iter=1000, random_state=SEED)
    l1.fit(train_F, ytr)
    cols = np.argsort(np.abs(l1.coef_[0]))[::-1][:NF].copy()

    Xtr_s = train_F[:, cols]
    mu = Xtr_s.mean(0); sd = Xtr_s.std(0); sd[sd == 0] = 1.0
    Xtr_z = (Xtr_s - mu) / sd

    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=300, random_state=SEED)
    clf.fit(Xtr_z, ytr)
    ptr = clf.predict_proba(Xtr_z)[:, 1]
    t = best_thresh(ytr, ptr)

    for name, (XF, L) in splits.items():
        if name == "indist":
            continue
        y = L[C].numpy().astype(int)
        npos = int(y.sum())
        base = float(y.mean())
        Xz = (XF[:, cols] - mu) / sd
        p = clf.predict_proba(Xz)[:, 1]
        pred = (p >= t).astype(int)
        pr, rc, f1 = prf(y, pred)
        predpos = float(pred.mean())
        # always-positive baseline
        _, _, apf = prf(y, np.ones_like(y))
        no_ood_pos = npos < 3
        degenerate = (predpos >= 0.97) or ((f1 - apf) < 0.02 and base > 0.5)
        rows.append(dict(concept=C, split=name, npos=npos, base=base,
                         precision=pr, recall=rc, f1=f1, predpos=predpos,
                         ap_f1=apf, lift=f1 - apf,
                         no_ood_pos=no_ood_pos, degenerate=bool(degenerate)))
        if (not no_ood_pos):
            ood_f1_for_mean.append(f1)

mean_ood = float(np.mean(ood_f1_for_mean))

print(f"=== GEMMA RESEED VERIFICATION  seed={SEED}  recipe=l1|nf400|mlp ===\n")
hdr = (f"{'concept':<10}{'split':>6}{'npos':>6}{'base':>7}{'prec':>7}"
       f"{'rec':>7}{'F1':>7}{'ppos%':>7}{'AP_F1':>7}{'lift':>7}  flag")
print(hdr); print("-" * len(hdr))
for r in rows:
    flag = ""
    if r["no_ood_pos"]: flag = "NO_OOD_POS"
    elif r["degenerate"]: flag = "DEGENERATE"
    print(f"{r['concept']:<10}{r['split']:>6}{r['npos']:>6}{r['base']:>7.3f}"
          f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
          f"{r['predpos']*100:>6.1f}%{r['ap_f1']:>7.3f}{r['lift']:>+7.3f}  {flag}")
print("-" * len(hdr))
print(f"mean OOD F1 (recipe-style, drops <3-pos splits) = {mean_ood:.4f}")
print("RESULT_JSON", json.dumps({"seed": SEED, "mean_ood_f1": mean_ood, "rows": rows}))
