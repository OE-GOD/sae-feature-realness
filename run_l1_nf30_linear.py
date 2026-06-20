import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 30

train_F = D["train_F"]
test_F  = D["test_F"]
ts_F    = D["ts_F"]
wk_F    = D["wk_F"]

def best_thresh_f1(probs, y):
    # tune threshold on these probs for best F1
    order = np.argsort(probs)
    cand = np.unique(probs)
    # evaluate a reasonable set of thresholds
    ths = np.unique(np.concatenate([cand, [0.0,1.0]]))
    if len(ths) > 2000:
        ths = np.quantile(probs, np.linspace(0,1,2000))
        ths = np.unique(ths)
    bestf, bestt = -1, 0.5
    for t in ths:
        pred = (probs >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > bestf:
            bestf, bestt = f, t
    return bestt

results = {}
indist_list = []
ood_vals = []

for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    Xtr = train_F.float().numpy()

    # 1. SELECT via L1 logreg on all 16384, top-k |weight|
    Xtr_full_std_mean = Xtr.mean(axis=0)
    Xtr_full_std_std  = Xtr.std(axis=0) + 1e-8
    Xtr_full_std = (Xtr - Xtr_full_std_mean) / Xtr_full_std_std
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000)
    l1.fit(Xtr_full_std, ytr)
    w = np.abs(l1.coef_.ravel())
    cols = np.argsort(w)[::-1][:NF]

    # 2. Standardize selected cols using train mean/std
    mean = Xtr[:, cols].mean(axis=0)
    std  = Xtr[:, cols].std(axis=0) + 1e-8
    Xtr_sel = (Xtr[:, cols] - mean) / std

    # 3. Train linear probe
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr_sel, ytr)

    # 4. Tune threshold on TRAIN
    ptr = clf.predict_proba(Xtr_sel)[:,1]
    thr = best_thresh_f1(ptr, ytr)

    def eval_split(F, Lmap, key, ood=False):
        y = Lmap[key].numpy().astype(int)
        if ood and y.sum() < 3:
            return None
        X = F.float().numpy()[:, cols]
        Xs = (X - mean) / std
        p = clf.predict_proba(Xs)[:,1]
        pred = (p >= thr).astype(int)
        return f1_score(y, pred, zero_division=0)

    indist = eval_split(test_F, D["test_L"], C, ood=False)
    ts = eval_split(ts_F, D["ts_L"], C, ood=True)
    wk = eval_split(wk_F, D["wk_L"], C, ood=True)

    results[C] = {"indist_f1": indist, "ts_f1": ts, "wk_f1": wk}
    indist_list.append(indist)
    for v in (ts, wk):
        if v is not None:
            ood_vals.append(v)

    del Xtr, Xtr_full_std, Xtr_sel
    print(C, results[C])

mean_indist = float(np.mean(indist_list))
mean_ood = float(np.mean(ood_vals))
print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)
import json
print("JSON", json.dumps({"mean_indist_f1":mean_indist,"mean_ood_f1":mean_ood,"per_concept":results}))
