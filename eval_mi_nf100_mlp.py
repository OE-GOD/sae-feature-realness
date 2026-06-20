import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0)
np.random.seed(0)

N_FEATURES = 100
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]

d = torch.load('/Users/oe/rebuild/gemma_detector_dataset.pt', map_location='cpu')
train_F = d['train_F'].float().numpy()
test_F  = d['test_F'].float().numpy()
ts_F    = d['ts_F'].float().numpy()
wk_F    = d['wk_F'].float().numpy()

def best_thresh_f1(y_true, scores):
    # tune threshold on these scores for best F1
    order = np.argsort(scores)
    cand = np.unique(scores)
    # limit candidates for speed but include enough resolution
    if len(cand) > 2000:
        cand = np.quantile(scores, np.linspace(0,1,2000))
        cand = np.unique(cand)
    best_f1, best_t = 0.0, 0.5
    for t in cand:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

results = {}
for C in CONCEPTS:
    ytr = d['train_L'][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue

    # 1. SELECT top-k mutual info, binarized feat>0
    Xbin = (train_F > 0).astype(int)
    mi = mutual_info_classif(Xbin, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:N_FEATURES]

    Xtr = train_F[:, cols]
    # 2. standardize using train mean/std
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd

    # 3. train MLP probe
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation='relu',
                        alpha=1e-3, max_iter=500, random_state=0)
    clf.fit(Xtr_s, ytr)

    # 4. tune threshold on TRAIN
    str_scores = clf.predict_proba(Xtr_s)[:, 1]
    thr, _ = best_thresh_f1(ytr, str_scores)

    def eval_split(Xsplit, ysplit):
        ysplit = ysplit.astype(int)
        Xs = (Xsplit[:, cols] - mu) / sd
        sc = clf.predict_proba(Xs)[:, 1]
        pred = (sc >= thr).astype(int)
        return f1_score(ysplit, pred, zero_division=0)

    indist_f1 = eval_split(test_F, d['test_L'][C].numpy())

    rec = {"concept": C, "indist_f1": float(indist_f1)}

    ts_y = d['ts_L'][C].numpy().astype(int)
    if ts_y.sum() >= 3:
        rec["ts_f1"] = float(eval_split(ts_F, ts_y))
    wk_y = d['wk_L'][C].numpy().astype(int)
    if wk_y.sum() >= 3:
        rec["wk_f1"] = float(eval_split(wk_F, wk_y))

    results[C] = rec
    print(rec, flush=True)

# aggregate
indist_vals = [r["indist_f1"] for r in results.values()]
ood_vals = []
for r in results.values():
    if "ts_f1" in r: ood_vals.append(r["ts_f1"])
    if "wk_f1" in r: ood_vals.append(r["wk_f1"])

print("MEAN_INDIST", float(np.mean(indist_vals)))
print("MEAN_OOD", float(np.mean(ood_vals)))

import json
print("JSON", json.dumps(list(results.values())))
