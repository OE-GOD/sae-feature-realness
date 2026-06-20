import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATA = "/Users/oe/rebuild/gemma_detector_dataset.pt"
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]
N_FEATURES = 100

d = torch.load(DATA, map_location="cpu")

train_F = d["train_F"].float().numpy()
test_F = d["test_F"].float().numpy()
ts_F = d["ts_F"].float().numpy()
wk_F = d["wk_F"].float().numpy()

def best_threshold(probs, y):
    # tune threshold for best F1 on train
    order = np.argsort(probs)
    cands = np.unique(probs)
    best_t, best_f = 0.5, -1.0
    # use a grid of candidate thresholds from unique probs
    grid = np.quantile(probs, np.linspace(0.0, 1.0, 201))
    grid = np.unique(np.concatenate([grid, [0.5]]))
    for t in grid:
        pred = (probs >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

results = []
indist_list = []
ood_list = []

for C in CONCEPTS:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue

    # 1. SELECT via L1 logreg on all 16384 features, top-k |weight|
    Xtr_full = train_F
    mean_full = Xtr_full.mean(axis=0)
    std_full = Xtr_full.std(axis=0) + 1e-8
    Xtr_std_full = (Xtr_full - mean_full) / std_full
    l1 = LogisticRegression(solver="liblinear", l1_ratio=1, C=1.0, max_iter=1000)
    l1.fit(Xtr_std_full, ytr)
    coef = np.abs(l1.coef_.ravel())
    cols = np.argsort(coef)[::-1][:N_FEATURES]

    # 2. Standardize selected cols using train mean/std
    mean_s = train_F[:, cols].mean(axis=0)
    std_s = train_F[:, cols].std(axis=0) + 1e-8
    Xtr = (train_F[:, cols] - mean_s) / std_s

    # 3. Train linear probe
    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(Xtr, ytr)

    # 4. Tune threshold on TRAIN
    ptr = probe.predict_proba(Xtr)[:, 1]
    thr = best_threshold(ptr, ytr)

    def eval_split(F, Ldict, min_pos=3, is_indist=False):
        y = Ldict[C].numpy().astype(int)
        if not is_indist and y.sum() < min_pos:
            return None
        X = (F[:, cols] - mean_s) / std_s
        p = probe.predict_proba(X)[:, 1]
        pred = (p >= thr).astype(int)
        return f1_score(y, pred, zero_division=0)

    indist_f1 = eval_split(test_F, d["test_L"], is_indist=True)
    ts_f1 = eval_split(ts_F, d["ts_L"])
    wk_f1 = eval_split(wk_F, d["wk_L"])

    rec = {"concept": C, "indist_f1": float(indist_f1)}
    indist_list.append(indist_f1)
    if ts_f1 is not None:
        rec["ts_f1"] = float(ts_f1)
        ood_list.append(ts_f1)
    if wk_f1 is not None:
        rec["wk_f1"] = float(wk_f1)
        ood_list.append(wk_f1)
    results.append(rec)
    print(C, rec, "thr=%.4f" % thr, "pos_train=%d" % ytr.sum())

mean_indist = float(np.mean(indist_list))
mean_ood = float(np.mean(ood_list))
print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)

import json
print("JSON_RESULTS", json.dumps(results))
