import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

d = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 400

train_F = d["train_F"].float().numpy()
test_F  = d["test_F"].float().numpy()
ts_F    = d["ts_F"].float().numpy()
wk_F    = d["wk_F"].float().numpy()

def best_thr(y, scores):
    order = np.argsort(scores)
    s = scores[order]; yy = y[order]
    # candidate thresholds
    thrs = np.unique(s)
    best_f1, best_t = 0.0, 0.5
    # subsample candidates if too many
    if len(thrs) > 500:
        thrs = np.quantile(s, np.linspace(0,1,500))
    for t in thrs:
        pred = (scores >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

per = []
indist_list, ood_list = [], []
for C in concepts:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    # 1. MI selection (binarize feat>0)
    Xb = (train_F > 0).astype(np.float32)
    mi = mutual_info_classif(Xb, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:NF]
    # 2. standardize using train mean/std
    Xtr = train_F[:, cols]
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1.0
    Xtr_s = (Xtr - mu)/sd
    # 3. probe
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_s, ytr)
    scores_tr = clf.decision_function(Xtr_s)
    # 4. threshold on train
    thr = best_thr(ytr, scores_tr)
    def evalF1(F, L):
        Xs = (F[:, cols]-mu)/sd
        sc = clf.decision_function(Xs)
        return f1_score(L, (sc>=thr).astype(int), zero_division=0)
    indist = evalF1(test_F, d["test_L"][C].numpy().astype(int))
    rec = {"concept": C, "indist_f1": float(indist)}
    indist_list.append(indist)
    ts_y = d["ts_L"][C].numpy().astype(int)
    if ts_y.sum() >= 3:
        v = evalF1(ts_F, ts_y); rec["ts_f1"]=float(v); ood_list.append(v)
    wk_y = d["wk_L"][C].numpy().astype(int)
    if wk_y.sum() >= 3:
        v = evalF1(wk_F, wk_y); rec["wk_f1"]=float(v); ood_list.append(v)
    per.append(rec)
    print(rec, flush=True)

print("RECIPE mi|nf400|linear")
print("mean_indist_f1", float(np.mean(indist_list)))
print("mean_ood_f1", float(np.mean(ood_list)))
import json; print("JSON", json.dumps(per))
