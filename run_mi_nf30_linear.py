import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 30

def npf(t): return t.numpy().astype(np.float32)

train_F = npf(D["train_F"])
test_F  = npf(D["test_F"])
ts_F    = npf(D["ts_F"])
wk_F    = npf(D["wk_F"])

def best_thr(probs, y):
    thrs = np.unique(probs)
    if len(thrs) > 512:
        thrs = np.quantile(probs, np.linspace(0,1,512))
    best_t, best_f = 0.5, -1
    for t in thrs:
        f = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f > best_f: best_f, best_t = f, t
    return best_t

results = []
indist_list, ood_list = [], []
for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    # 1. MI selection (binarized feat>0)
    Xbin = (train_F > 0).astype(np.float32)
    mi = mutual_info_classif(Xbin, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:NF]
    # 2. standardize using train mean/std
    mu = train_F[:, cols].mean(0)
    sd = train_F[:, cols].std(0) + 1e-8
    Xtr = (train_F[:, cols] - mu) / sd
    # 3. probe
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    # 4. threshold on train
    ptr = clf.predict_proba(Xtr)[:,1]
    thr = best_thr(ptr, ytr)
    # 5. eval
    def ev(F, L, need_pos=False):
        y = L[C].numpy().astype(int)
        if need_pos and y.sum() < 3:
            return None
        X = (F[:, cols] - mu) / sd
        p = clf.predict_proba(X)[:,1]
        return f1_score(y, (p >= thr).astype(int), zero_division=0)

    indist = ev(test_F, D["test_L"])
    ts = ev(ts_F, D["ts_L"], need_pos=True)
    wk = ev(wk_F, D["wk_L"], need_pos=True)
    rec = {"concept": C, "indist_f1": float(indist)}
    if ts is not None: rec["ts_f1"] = float(ts)
    if wk is not None: rec["wk_f1"] = float(wk)
    results.append(rec)
    indist_list.append(indist)
    for v in (ts, wk):
        if v is not None: ood_list.append(v)
    print(rec)

print("MEAN_INDIST", float(np.mean(indist_list)))
print("MEAN_OOD", float(np.mean(ood_list)))
import json; print("JSON", json.dumps(results))
