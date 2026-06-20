import torch, numpy as np, json, gc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)

D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]

train_F = D["train_F"].float().numpy()
test_F  = D["test_F"].float().numpy()
ts_F    = D["ts_F"].float().numpy()
wk_F    = D["wk_F"].float().numpy()

NF = 400

# global standardization params for L1 selection (fit on train)
g_mu = train_F.mean(0); g_sd = train_F.std(0); g_sd[g_sd==0] = 1.0
train_std = (train_F - g_mu) / g_sd

def best_thresh(y, scores):
    cand = np.unique(scores)
    mids = (cand[:-1]+cand[1:])/2 if len(cand)>1 else np.array([])
    ths = np.concatenate([[scores.min()-1e-6], mids, [scores.max()+1e-6]])
    bf, bt = -1, 0.5
    for t in ths:
        f = f1_score(y, (scores>=t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return bt

results = []
for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    # 1. SELECT via L1-logreg on all 16384 standardized features
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=500)
    l1.fit(train_std, ytr)
    w = np.abs(l1.coef_[0])
    cols = np.argsort(w)[::-1][:NF]

    # 2. standardize selected cols using train mean/std
    mu = train_F[:, cols].mean(0); sd = train_F[:, cols].std(0); sd[sd==0]=1.0
    Xtr = (train_F[:, cols] - mu) / sd

    # 3. linear probe
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)

    # 4. tune threshold on train
    str_tr = clf.decision_function(Xtr)
    th = best_thresh(ytr, str_tr)

    def ev(F, L):
        y = L[C].numpy().astype(int)
        Xn = (F[:, cols] - mu) / sd
        s = clf.decision_function(Xn)
        return y, s

    yte, ste = ev(test_F, D["test_L"])
    res = {"concept": C, "indist_f1": float(f1_score(yte,(ste>=th).astype(int),zero_division=0))}
    yts, sts = ev(ts_F, D["ts_L"])
    if yts.sum() >= 3:
        res["ts_f1"] = float(f1_score(yts,(sts>=th).astype(int),zero_division=0))
    ywk, swk = ev(wk_F, D["wk_L"])
    if ywk.sum() >= 3:
        res["wk_f1"] = float(f1_score(ywk,(swk>=th).astype(int),zero_division=0))
    results.append(res)
    print(res, "ts_pos", int(yts.sum()), "wk_pos", int(ywk.sum()), flush=True)
    del Xtr, l1, clf
    gc.collect()

mean_indist = float(np.mean([r["indist_f1"] for r in results]))
ood_vals = [r[k] for r in results for k in ("ts_f1","wk_f1") if k in r]
mean_ood = float(np.mean(ood_vals))
out = {"recipe":"l1|nf400|linear","mean_indist_f1":mean_indist,
       "mean_ood_f1":mean_ood,"per_concept":results}
print("RESULT", json.dumps(out))
with open("/Users/oe/rebuild/recipe_l1_nf400_linear.json","w") as f:
    json.dump(out, f)
