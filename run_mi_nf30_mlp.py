import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)
D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 30

train_F = D["train_F"]  # [9118,16384] f16
test_F  = D["test_F"]; ts_F = D["ts_F"]; wk_F = D["wk_F"]

def best_thresh_f1(y, p):
    thr = np.unique(p)
    if len(thr) > 512:
        thr = np.quantile(p, np.linspace(0,1,512))
    best_t, best_f = 0.5, -1
    for t in thr:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f: best_f, best_t = f, t
    return best_t, best_f

def eval_split(F, L, C, cols, mean, std, clf, thr):
    y = L[C].numpy().astype(int)
    if y.sum() < 3:
        return None
    X = F[:, cols].float().numpy()
    X = (X - mean) / std
    p = clf.predict_proba(X)[:,1]
    return f1_score(y, (p >= thr).astype(int), zero_division=0)

results = []
for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    Xtr_full = train_F.float().numpy()
    # MI on binarized feat>0
    Xbin = (Xtr_full > 0).astype(int)
    mi = mutual_info_classif(Xbin, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:NF].copy()
    Xsel = Xtr_full[:, cols]
    mean = Xsel.mean(0); std = Xsel.std(0); std[std==0]=1.0
    Xs = (Xsel - mean)/std
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=300, random_state=0)
    clf.fit(Xs, ytr)
    ptr = clf.predict_proba(Xs)[:,1]
    thr, _ = best_thresh_f1(ytr, ptr)

    ind = eval_split(test_F, D["test_L"], C, cols, mean, std, clf, thr)
    ts  = eval_split(ts_F,   D["ts_L"],   C, cols, mean, std, clf, thr)
    wk  = eval_split(wk_F,   D["wk_L"],   C, cols, mean, std, clf, thr)
    results.append((C, ind, ts, wk))
    print(C, "ind=%.4f"%ind, "ts=%s"%(("%.4f"%ts) if ts is not None else "skip"),
          "wk=%s"%(("%.4f"%wk) if wk is not None else "skip"))
    del Xtr_full, Xbin, Xsel, Xs

indist = [r[1] for r in results]
ood = []
for r in results:
    if r[2] is not None: ood.append(r[2])
    if r[3] is not None: ood.append(r[3])
print("MEAN_INDIST", np.mean(indist))
print("MEAN_OOD", np.mean(ood))
import json
print("JSON", json.dumps([{"concept":r[0],"indist_f1":r[1],"ts_f1":r[2],"wk_f1":r[3]} for r in results]))
