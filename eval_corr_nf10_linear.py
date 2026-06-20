import torch, numpy as np, json

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float().numpy(); teF = d["test_F"].float().numpy(); ooF = d["ood_F"].float().numpy()

NF = 10

def select_corr(F, y, k):
    yc = y - y.mean()
    Fc = F - F.mean(0)
    num = (Fc * yc[:, None]).sum(0)
    den = np.sqrt((Fc**2).sum(0) * (yc**2).sum() + 1e-12)
    r = np.abs(num / den)
    return np.argsort(-r)[:k]

def f1_score(y, p):
    tp = np.sum((p == 1) & (y == 1)); fp = np.sum((p == 1) & (y == 0)); fn = np.sum((p == 0) & (y == 1))
    if tp == 0: return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def train_logreg(X, y, lr=0.5, iters=2000, l2=1e-4):
    n, dd = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(dd + 1)
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    return w

def predict_proba(X, w):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return sigmoid(Xb @ w)

def best_thresh(probs, y):
    bestf, bestt = -1, 0.5
    for t in np.unique(probs):
        f = f1_score(y, (probs >= t).astype(int))
        if f > bestf: bestf, bestt = f, t
    return bestt

rows = []
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5: continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)
    cols = select_corr(trF, ytr.astype(float), NF)
    Xtr = trF[:, cols]; mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (teF[:, cols] - mu) / sd; Xoo = (ooF[:, cols] - mu) / sd
    w = train_logreg(Xtr, ytr.astype(float))
    ptr = predict_proba(Xtr, w)
    t = best_thresh(ptr, ytr)
    f_te = f1_score(yte, (predict_proba(Xte, w) >= t).astype(int))
    f_oo = f1_score(yoo, (predict_proba(Xoo, w) >= t).astype(int))
    rows.append((C, float(f_te), float(f_oo)))
    print(C, f_te, f_oo)

mte = np.mean([r[1] for r in rows]); moo = np.mean([r[2] for r in rows])
print("MEAN_INDIST", mte); print("MEAN_OOD", moo)
print(json.dumps({"mean_indist": float(mte), "mean_ood": float(moo),
    "per": [{"concept": c, "indist_f1": a, "ood_f1": b} for c, a, b in rows]}))
