import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATA = "/Users/oe/rebuild/gemma_detector_dataset.pt"
N_FEATURES = 200
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]

d = torch.load(DATA, map_location="cpu")

def best_threshold(scores, y):
    # scores: probability of positive
    order = np.argsort(scores)
    # candidate thresholds = unique scores
    thr_cands = np.unique(scores)
    best_t, best_f1 = 0.5, -1.0
    # limit candidates for speed
    if len(thr_cands) > 500:
        thr_cands = np.quantile(scores, np.linspace(0,1,500))
    for t in thr_cands:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

train_F = d["train_F"]
test_F = d["test_F"]
ts_F = d["ts_F"]
wk_F = d["wk_F"]

results = []
indist_vals = []
ood_vals = []

for C in CONCEPTS:
    ytr = np.asarray(d["train_L"][C]).astype(int)
    if ytr.sum() < 5:
        continue

    Xtr = train_F.float().numpy()

    # 1. SELECT via L1 logreg on all 16384 features, top-k |weight|
    # standardize first for L1 selection? Procedure says standardize at step 2 (after select).
    # L1 selection done on raw train_F. Use standardization for stable L1 fit.
    mu_all = Xtr.mean(0)
    sd_all = Xtr.std(0) + 1e-8
    Xtr_std_all = (Xtr - mu_all) / sd_all

    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000)
    l1.fit(Xtr_std_all, ytr)
    w = np.abs(l1.coef_.ravel())
    cols = np.argsort(w)[::-1][:N_FEATURES]

    # 2. Standardize selected cols using train mean/std
    Xtr_sel = Xtr[:, cols]
    mu = Xtr_sel.mean(0)
    sd = Xtr_sel.std(0) + 1e-8
    Xtr_sel_std = (Xtr_sel - mu) / sd

    # 3. Train linear probe
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr_sel_std, ytr)

    # 4. Tune threshold on train for best F1
    tr_scores = clf.predict_proba(Xtr_sel_std)[:, 1]
    thr = best_threshold(tr_scores, ytr)

    def eval_split(F, L):
        y = np.asarray(L).astype(int)
        X = F.float().numpy()[:, cols]
        Xs = (X - mu) / sd
        s = clf.predict_proba(Xs)[:, 1]
        pred = (s >= thr).astype(int)
        return f1_score(y, pred, zero_division=0), int(y.sum())

    indist_f1, _ = eval_split(test_F, d["test_L"][C])

    ts_y = np.asarray(d["ts_L"][C]).astype(int)
    wk_y = np.asarray(d["wk_L"][C]).astype(int)

    entry = {"concept": C, "indist_f1": float(indist_f1)}
    indist_vals.append(indist_f1)

    ood_this = []
    if ts_y.sum() >= 3:
        ts_f1, _ = eval_split(ts_F, d["ts_L"][C])
        entry["ts_f1"] = float(ts_f1)
        ood_this.append(ts_f1)
    if wk_y.sum() >= 3:
        wk_f1, _ = eval_split(wk_F, d["wk_L"][C])
        entry["wk_f1"] = float(wk_f1)
        ood_this.append(wk_f1)

    ood_vals.extend(ood_this)
    results.append(entry)
    print(C, entry, flush=True)

    del Xtr, Xtr_std_all, Xtr_sel, Xtr_sel_std

mean_indist = float(np.mean(indist_vals)) if indist_vals else 0.0
mean_ood = float(np.mean(ood_vals)) if ood_vals else 0.0

print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)

import json
print("JSON_RESULT", json.dumps({
    "recipe": "l1|nf200|linear",
    "mean_indist_f1": mean_indist,
    "mean_ood_f1": mean_ood,
    "per_concept": results
}))
