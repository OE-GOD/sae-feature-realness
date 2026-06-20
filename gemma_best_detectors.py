"""
gemma_best_detectors.py
=======================
Builds the BEST detector per concept on the REAL Gemma Scope SAE using the
WINNING recipe from the detector-recipe sweep, ranked by mean cross-distribution
(OOD = avg of TinyStories + wikitext) F1:

    WINNER:  l1 | nf400 | mlp     (mean OOD F1 = 0.98026)

Recipe = feature SELECTION via L1-regularized logistic regression over all
16384 Gemma Scope SAE features -> keep top 400 by |weight| -> standardize those
400 features with TRAIN mean/std -> train a small MLP (32 hidden, relu) ->
pick the decision threshold that maximizes F1 on the TRAIN set -> evaluate on
in-dist test and the two OOD splits (TinyStories, wikitext).

This reproduces the procedure used by run_l1_nf400_mlp.py (the sweep's winning
recipe). For each concept it prints in-dist + both OOD F1s and saves every
trained detector (sklearn MLP, selected feature columns, normalization stats,
threshold) to gemma_best_detectors.pt.
"""

import torch, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)

DATA = "/Users/oe/rebuild/gemma_detector_dataset.pt"
OUT = "/Users/oe/rebuild/gemma_best_detectors.pt"
RECIPE = "l1|nf400|mlp"
NF = 400  # number of features kept by the L1 selector

d = torch.load(DATA, map_location="cpu")
CONCEPTS = d["keys"]
train_F = d["train_F"].float()
splits = {
    "indist": (d["test_F"].float(), d["test_L"]),
    "ts":     (d["ts_F"].float(),   d["ts_L"]),
    "wk":     (d["wk_F"].float(),   d["wk_L"]),
}


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


print(f"WINNING RECIPE: {RECIPE}  (NF={NF})\n")
print(f"{'concept':<11} {'in-dist':>8} {'ts(OOD)':>9} {'wk(OOD)':>9}")
print("-" * 40)

detectors = {}
per_concept = []
indist_vals, ood_vals = [], []

for C in CONCEPTS:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        print(f"skip {C}: <5 train pos")
        continue

    Xtr = train_F.numpy()

    # 1. L1 selection over all 16384 features
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000)
    l1.fit(Xtr, ytr)
    w = np.abs(l1.coef_[0])
    cols = np.argsort(w)[::-1][:NF].copy()

    Xtr_s = Xtr[:, cols]
    # 2. standardize with train mean/std
    mu = Xtr_s.mean(0); sd = Xtr_s.std(0); sd[sd == 0] = 1.0
    Xtr_z = (Xtr_s - mu) / sd

    # 3. MLP probe
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=300, random_state=0)
    clf.fit(Xtr_z, ytr)

    # 4. threshold tuned on train
    ptr = clf.predict_proba(Xtr_z)[:, 1]
    t = best_thresh(ytr, ptr)

    rec = {"concept": C, "indist_f1": None, "ts_f1": None, "wk_f1": None}
    for name, (XF, L) in splits.items():
        y = L[C].numpy().astype(int)
        if name != "indist" and y.sum() < 3:
            rec[f"{name}_f1"] = None  # no/too-few OOD positives in this split
            continue
        Xz = (XF.numpy()[:, cols] - mu) / sd
        p = clf.predict_proba(Xz)[:, 1]
        rec[f"{name}_f1"] = float(f1_score(y, (p >= t).astype(int), zero_division=0))

    def fmt(v):
        return f"{v:>9.4f}" if v is not None else f"{'n/a':>9}"
    print(f"{C:<11} {rec['indist_f1']:>8.4f} {fmt(rec['ts_f1'])} {fmt(rec['wk_f1'])}")

    indist_vals.append(rec["indist_f1"])
    for nm in ("ts_f1", "wk_f1"):
        if rec[nm] is not None:
            ood_vals.append(rec[nm])

    out_rec = {"concept": C, "indist_f1": rec["indist_f1"]}
    if rec["ts_f1"] is not None: out_rec["ts_f1"] = rec["ts_f1"]
    if rec["wk_f1"] is not None: out_rec["wk_f1"] = rec["wk_f1"]
    per_concept.append(out_rec)

    detectors[C] = {
        "cols": torch.from_numpy(cols),       # indices into 16384-d SAE space
        "mu": torch.from_numpy(mu),           # per-feature train mean
        "sd": torch.from_numpy(sd),           # per-feature train std
        "threshold": float(t),                # F1-optimal threshold (on train)
        "mlp": clf,                           # fitted sklearn MLPClassifier
        "indist_f1": rec["indist_f1"],
        "ts_f1": rec["ts_f1"],
        "wk_f1": rec["wk_f1"],
    }

mte = float(np.mean(indist_vals))
moo = float(np.mean(ood_vals))
print("-" * 40)
print(f"MEAN in-dist F1: {mte:.4f}")
print(f"MEAN OOD F1 (avg of all ts+wk): {moo:.4f}")

torch.save({
    "recipe": RECIPE,
    "nf": NF,
    "keys": list(detectors.keys()),
    "detectors": detectors,
    "mean_indist_f1": mte,
    "mean_ood_f1": moo,
}, OUT)
print(f"\nSaved {len(detectors)} detectors -> {OUT}")
print("RESULT_JSON", json.dumps({
    "recipe": RECIPE,
    "mean_indist_f1": mte,
    "mean_ood_f1": moo,
    "per_concept": per_concept,
}))
