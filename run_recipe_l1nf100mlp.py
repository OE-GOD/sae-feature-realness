import torch, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)
D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]

train_F = D["train_F"].float().numpy()
test_F = D["test_F"].float().numpy()
ts_F = D["ts_F"].float().numpy()
wk_F = D["wk_F"].float().numpy()
NF = 100

def best_thr_f1(y, scores):
    thr_cands = np.unique(scores)
    if len(thr_cands) > 500:
        thr_cands = np.quantile(scores, np.linspace(0,1,500))
    best_f1, best_t = -1.0, 0.5
    for t in thr_cands:
        f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

results = {}
for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    sel_model = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=200)
    sel_model.fit(train_F, ytr)
    w = np.abs(sel_model.coef_.ravel())
    cols = np.argsort(w)[::-1][:NF]
    Xtr = train_F[:, cols]
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    probe = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-4,
                          max_iter=300, random_state=0)
    probe.fit(Xtr, ytr)
    str_scores = probe.predict_proba(Xtr)[:, 1]
    thr = best_thr_f1(ytr, str_scores)

    def eval_split(F, L):
        y = L.numpy().astype(int)
        X = (F[:, cols] - mu) / sd
        s = probe.predict_proba(X)[:, 1]
        return y, s

    yte, ste = eval_split(test_F, D["test_L"][C])
    indist = f1_score(yte, (ste >= thr).astype(int), zero_division=0)
    entry = {"concept": C, "indist_f1": float(indist)}
    yts, sts = eval_split(ts_F, D["ts_L"][C])
    if yts.sum() >= 3:
        entry["ts_f1"] = float(f1_score(yts, (sts >= thr).astype(int), zero_division=0))
    ywk, swk = eval_split(wk_F, D["wk_L"][C])
    if ywk.sum() >= 3:
        entry["wk_f1"] = float(f1_score(ywk, (swk >= thr).astype(int), zero_division=0))
    results[C] = entry
    print(entry, flush=True)

per = list(results.values())
mean_indist = float(np.mean([r["indist_f1"] for r in per]))
ood_vals = []
for r in per:
    if "ts_f1" in r: ood_vals.append(r["ts_f1"])
    if "wk_f1" in r: ood_vals.append(r["wk_f1"])
mean_ood = float(np.mean(ood_vals))
print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)
print("JSON", json.dumps(per))
