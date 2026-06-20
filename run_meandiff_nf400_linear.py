import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]
N_FEATURES = 400

d = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")

train_F = d["train_F"].float().numpy()
test_F = d["test_F"].float().numpy()
ts_F = d["ts_F"].float().numpy()
wk_F = d["wk_F"].float().numpy()

def best_threshold(scores, y):
    # find threshold maximizing F1 on given scores
    order = np.argsort(scores)
    s_sorted = scores[order]
    # candidate thresholds: midpoints + extremes
    cands = np.unique(s_sorted)
    if len(cands) > 2000:
        cands = np.quantile(scores, np.linspace(0,1,2000))
    best_f1, best_t = -1, 0.5
    for t in cands:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

results = {}
indist_list = []
ood_list = []

for C in CONCEPTS:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue

    # 1. SELECT meandiff top-k
    pos = train_F[ytr==1]
    neg = train_F[ytr==0]
    md = np.abs(pos.mean(0) - neg.mean(0))
    cols = np.argsort(md)[::-1][:N_FEATURES]

    Xtr = train_F[:, cols]
    # 2. standardize using train mean/std
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd

    # 3. train linear probe
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_s, ytr)

    # 4. tune threshold on TRAIN
    tr_scores = clf.decision_function(Xtr_s)
    thr = best_threshold(tr_scores, ytr)

    def eval_split(F, y):
        Xs = (F[:, cols] - mu) / sd
        sc = clf.decision_function(Xs)
        pred = (sc >= thr).astype(int)
        return f1_score(y, pred, zero_division=0)

    yte = d["test_L"][C].numpy().astype(int)
    indist_f1 = eval_split(test_F, yte)

    entry = {"concept": C, "indist_f1": float(indist_f1)}
    indist_list.append(indist_f1)

    yts = d["ts_L"][C].numpy().astype(int)
    if yts.sum() >= 3:
        ts_f1 = eval_split(ts_F, yts)
        entry["ts_f1"] = float(ts_f1)
        ood_list.append(ts_f1)

    ywk = d["wk_L"][C].numpy().astype(int)
    if ywk.sum() >= 3:
        wk_f1 = eval_split(wk_F, ywk)
        entry["wk_f1"] = float(wk_f1)
        ood_list.append(wk_f1)

    results[C] = entry
    print(entry)

mean_indist = float(np.mean(indist_list))
mean_ood = float(np.mean(ood_list))
print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)

import json
print("JSON_RESULT", json.dumps({"per_concept": list(results.values()),
                                 "mean_indist_f1": mean_indist,
                                 "mean_ood_f1": mean_ood}))
