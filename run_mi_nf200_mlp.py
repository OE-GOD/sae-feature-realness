import torch, numpy as np, gc
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)
NF = 200
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]

D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
train_F = D["train_F"].float().numpy()
test_F  = D["test_F"].float().numpy()
ts_F    = D["ts_F"].float().numpy()
wk_F    = D["wk_F"].float().numpy()
train_L = D["train_L"]; test_L = D["test_L"]; ts_L = D["ts_L"]; wk_L = D["wk_L"]
del D; gc.collect()

def best_thr(y, scores):
    order = np.argsort(scores)
    s_sorted = scores[order]
    cands = np.concatenate([[s_sorted[0]-1e-6], (s_sorted[:-1]+s_sorted[1:])/2, [s_sorted[-1]+1e-6]])
    bestf, bestt = -1, 0.0
    for t in cands:
        f = f1_score(y, (scores >= t).astype(int))
        if f > bestf: bestf, bestt = f, t
    return bestt

results = []
indist_vals, ood_vals = [], []

for C in CONCEPTS:
    ytr = np.asarray(train_L[C]).astype(int)
    if ytr.sum() < 5:
        continue
    # 1. MI selection on binarized features (feat>0)
    Xbin = (train_F > 0).astype(int)
    mi = mutual_info_classif(Xbin, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:NF]
    del Xbin, mi; gc.collect()

    Xtr = train_F[:, cols]
    mean = Xtr.mean(0); std = Xtr.std(0); std[std == 0] = 1.0
    Xtr_s = (Xtr - mean) / std

    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-3,
                        max_iter=300, random_state=0)
    clf.fit(Xtr_s, ytr)
    str_tr = clf.predict_proba(Xtr_s)[:, 1]
    thr = best_thr(ytr, str_tr)

    def ev(F, L, split):
        y = np.asarray(L[C]).astype(int)
        if split != "indist" and y.sum() < 3:
            return None
        Xs = (F[:, cols] - mean) / std
        sc = clf.predict_proba(Xs)[:, 1]
        return f1_score(y, (sc >= thr).astype(int))

    indist = ev(test_F, test_L, "indist")
    ts = ev(ts_F, ts_L, "ts")
    wk = ev(wk_F, wk_L, "wk")

    rec = {"concept": C, "indist_f1": float(indist)}
    indist_vals.append(indist)
    if ts is not None: rec["ts_f1"] = float(ts); ood_vals.append(ts)
    if wk is not None: rec["wk_f1"] = float(wk); ood_vals.append(wk)
    results.append(rec)
    print(rec, flush=True)

import json
out = {
  "recipe": "mi|nf200|mlp",
  "mean_indist_f1": float(np.mean(indist_vals)),
  "mean_ood_f1": float(np.mean(ood_vals)),
  "per_concept": results,
}
print("RESULT_JSON", json.dumps(out))
