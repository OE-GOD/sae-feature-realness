import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 30

train_F = D["train_F"].float().numpy()
test_F  = D["test_F"].float().numpy()
ts_F    = D["ts_F"].float().numpy()
wk_F    = D["wk_F"].float().numpy()

def best_thr(y, scores):
    order = np.argsort(scores)
    s = scores[order]; yy = y[order]
    # candidate thresholds = midpoints; brute force over unique
    thrs = np.unique(scores)
    best_f1, best_t = -1, 0.5
    # limit candidates for speed
    cand = thrs if len(thrs) <= 2000 else np.quantile(thrs, np.linspace(0,1,2000))
    for t in cand:
        pred = (scores >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

results = {}
for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    # 1. L1 selection on all 16384
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=1000)
    l1.fit(train_F, ytr)
    w = np.abs(l1.coef_.ravel())
    cols = np.argsort(w)[::-1][:NF]
    Xtr = train_F[:, cols]
    # 2. standardize using train mean/std
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1.0
    Xtr_s = (Xtr - mu)/sd
    # 3. MLP probe
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=500, random_state=0)
    clf.fit(Xtr_s, ytr)
    str_tr = clf.predict_proba(Xtr_s)[:,1]
    # 4. tune threshold on train
    thr = best_thr(ytr, str_tr)

    def eval_split(F, L, split):
        y = L[C].numpy().astype(int) if isinstance(L, dict) else L
        if y.sum() < 3 and split != "indist":
            return None
        Xs = (F[:, cols]-mu)/sd
        sc = clf.predict_proba(Xs)[:,1]
        return f1_score(y, (sc>=thr).astype(int), zero_division=0)

    indist = eval_split(test_F, D["test_L"], "indist")
    ts = eval_split(ts_F, D["ts_L"], "ts")
    wk = eval_split(wk_F, D["wk_L"], "wk")
    results[C] = (indist, ts, wk)
    print(C, indist, ts, wk, flush=True)

per = []
ind_vals = []
ood_vals = []
for C,(i,t,w) in results.items():
    d = {"concept":C, "indist_f1":float(i)}
    if t is not None: d["ts_f1"]=float(t); ood_vals.append(t)
    if w is not None: d["wk_f1"]=float(w); ood_vals.append(w)
    per.append(d)
    ind_vals.append(i)

import json
out = {"recipe":"l1|nf30|mlp",
       "mean_indist_f1":float(np.mean(ind_vals)),
       "mean_ood_f1":float(np.mean(ood_vals)),
       "per_concept":per}
print("RESULT_JSON")
print(json.dumps(out))
