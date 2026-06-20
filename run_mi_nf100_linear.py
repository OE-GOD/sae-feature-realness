import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

torch.manual_seed(0)
np.random.seed(0)

D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 100

train_F = D["train_F"]  # [9118,16384] float16

def best_thresh(y, scores):
    # try candidate thresholds from sorted scores
    order = np.argsort(scores)
    s_sorted = scores[order]
    cands = np.unique(s_sorted)
    # limit candidates for speed
    if len(cands) > 2000:
        cands = cands[np.linspace(0, len(cands)-1, 2000).astype(int)]
    best_f1, best_t = -1.0, 0.5
    for t in cands:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

results = {}
indist_list = []
ood_list = []

for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    Xtr = train_F.float().numpy()

    # 1. SELECT via mutual info (binarized feat>0)
    Xbin = (Xtr > 0).astype(int)
    mi = mutual_info_classif(Xbin, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:NF]

    Xtr_sel = Xtr[:, cols]
    # 2. standardize with train mean/std
    mu = Xtr_sel.mean(axis=0)
    sd = Xtr_sel.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr_std = (Xtr_sel - mu) / sd

    # 3. probe linear
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_std, ytr)

    # 4. tune threshold on train
    tr_scores = clf.predict_proba(Xtr_std)[:, 1]
    t, _ = best_thresh(ytr, tr_scores)

    rec = {}
    # in-dist
    def eval_split(Fkey, Lkey):
        X = D[Fkey].float().numpy()[:, cols]
        X = (X - mu) / sd
        y = D[Lkey][C].numpy().astype(int)
        sc = clf.predict_proba(X)[:, 1]
        pred = (sc >= t).astype(int)
        return f1_score(y, pred, zero_division=0), y.sum()

    indist_f1, _ = eval_split("test_F", "test_L")
    rec["concept"] = C
    rec["indist_f1"] = float(indist_f1)
    indist_list.append(indist_f1)

    ts_f1, ts_pos = eval_split("ts_F", "ts_L")
    if ts_pos >= 3:
        rec["ts_f1"] = float(ts_f1)
        ood_list.append(ts_f1)
    wk_f1, wk_pos = eval_split("wk_F", "wk_L")
    if wk_pos >= 3:
        rec["wk_f1"] = float(wk_f1)
        ood_list.append(wk_f1)

    results[C] = rec
    print(C, rec, "ts_pos", int(ts_pos), "wk_pos", int(wk_pos))
    del Xtr, Xbin, Xtr_sel, Xtr_std

mean_indist = float(np.mean(indist_list))
mean_ood = float(np.mean(ood_list))
print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)

import json
print("JSON_OUT", json.dumps({"mean_indist_f1": mean_indist, "mean_ood_f1": mean_ood,
    "per_concept": list(results.values())}))
