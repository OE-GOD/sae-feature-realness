import torch, numpy as np, json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)

D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]

train_F = D["train_F"]; train_L = D["train_L"]
test_F = D["test_F"]; test_L = D["test_L"]
ts_F = D["ts_F"]; ts_L = D["ts_L"]
wk_F = D["wk_F"]; wk_L = D["wk_L"]

NF = 400

def best_thresh(y, scores):
    cand = np.unique(scores)
    mids = (cand[:-1]+cand[1:])/2 if len(cand)>1 else np.array([])
    ths = np.concatenate([[scores.min()-1e-6], mids, [scores.max()+1e-6]])
    bf, bt = -1, 0.5
    for t in ths:
        f = f1_score(y, (scores>=t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return bt

train_all = train_F.float().numpy()
results = {}
for C in concepts:
    ytr = train_L[C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    pos = train_all[ytr==1]; neg = train_all[ytr==0]
    md = np.abs(pos.mean(0) - neg.mean(0))
    cols = np.argsort(md)[::-1][:NF]
    del pos, neg

    Xtr_s = train_all[:, cols]
    mu = Xtr_s.mean(0); sd = Xtr_s.std(0); sd[sd==0]=1.0
    Xtr_n = (Xtr_s - mu)/sd

    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=300, random_state=0)
    clf.fit(Xtr_n, ytr)
    th = best_thresh(ytr, clf.predict_proba(Xtr_n)[:,1])

    def ev(F, L):
        y = L[C].numpy().astype(int)
        Xn = (F.float().numpy()[:, cols]-mu)/sd
        s = clf.predict_proba(Xn)[:,1]
        return y, s

    yte, ste = ev(test_F, test_L)
    res = {"concept":C, "indist_f1":float(f1_score(yte,(ste>=th).astype(int),zero_division=0))}
    yts, sts = ev(ts_F, ts_L)
    if yts.sum() >= 3:
        res["ts_f1"] = float(f1_score(yts,(sts>=th).astype(int),zero_division=0))
    ywk, swk = ev(wk_F, wk_L)
    if ywk.sum() >= 3:
        res["wk_f1"] = float(f1_score(ywk,(swk>=th).astype(int),zero_division=0))
    results[C] = res
    print(res, flush=True)
    del Xtr_s, Xtr_n

per = list(results.values())
mean_indist = float(np.mean([r["indist_f1"] for r in per]))
ood_vals = [r[k] for r in per for k in ("ts_f1","wk_f1") if k in r]
mean_ood = float(np.mean(ood_vals))
print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)
print("JSON", json.dumps(per))
