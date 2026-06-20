import torch, math, json
import numpy as np

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float().numpy()
test_F = d["test_F"].float().numpy()
ood_F = d["ood_F"].float().numpy()

N_FEAT = 100

class MLP:
    def __init__(self, n_in, h=32, lr=0.05, iters=2000, l2=1e-4, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, h) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(h)
        self.W2 = rng.randn(h, 1) * np.sqrt(2.0 / h)
        self.b2 = 0.0
        self.lr = lr; self.iters = iters; self.l2 = l2

    def fit(self, X, y):
        n = X.shape[0]
        yf = y.astype(float).reshape(-1, 1)
        for _ in range(self.iters):
            z1 = X @ self.W1 + self.b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ self.W2 + self.b2
            p = 1 / (1 + np.exp(-z2))
            dz2 = (p - yf) / n
            gW2 = a1.T @ dz2 + self.l2 * self.W2
            gb2 = dz2.sum()
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)
            gW1 = X.T @ dz1 + self.l2 * self.W1
            gb1 = dz1.sum(0)
            self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
            self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1

    def proba(self, X):
        a1 = np.maximum(0, X @ self.W1 + self.b1)
        z2 = a1 @ self.W2 + self.b2
        return (1 / (1 + np.exp(-z2))).ravel()

def mi_select(X, y, k):
    n = len(y)
    yb = y.astype(bool)
    Xb = X > 0
    scores = np.zeros(X.shape[1])
    py = {True: yb.mean(), False: 1 - yb.mean()}
    for j in range(X.shape[1]):
        xb = Xb[:, j]
        px = {True: xb.mean(), False: 1 - xb.mean()}
        mi = 0.0
        for xv in (False, True):
            for yv in (False, True):
                cnt = np.sum((xb == xv) & (yb == yv))
                if cnt == 0:
                    continue
                pxy = cnt / n
                if px[xv] > 0 and py[yv] > 0:
                    mi += pxy * math.log(pxy / (px[xv] * py[yv]))
        scores[j] = mi
    return np.argsort(-scores)[:k]

def f1_score(yt, yp):
    tp = np.sum((yp == 1) & (yt == 1))
    fp = np.sum((yp == 1) & (yt == 0))
    fn = np.sum((yp == 0) & (yt == 1))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)

def best_thresh(scores, y):
    cand = np.unique(scores)
    bf, bt = 0.0, 0.5
    for t in cand:
        f = f1_score(y, (scores >= t).astype(int))
        if f > bf:
            bf, bt = f, t
    return bt

results = []
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].numpy().astype(int)
    yood = d["ood_L"][C].numpy().astype(int)

    cols = mi_select(train_F, ytr, N_FEAT)
    Xtr = train_F[:, cols]; Xte = test_F[:, cols]; Xood = ood_F[:, cols]
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd; Xood = (Xood - mu) / sd

    clf = MLP(Xtr.shape[1], h=32, lr=0.05, iters=2000, l2=1e-4, seed=0)
    clf.fit(Xtr, ytr)
    s_tr = clf.proba(Xtr)
    s_te = clf.proba(Xte)
    s_ood = clf.proba(Xood)

    t = best_thresh(s_tr, ytr)
    f_in = f1_score(yte, (s_te >= t).astype(int))
    f_ood = f1_score(yood, (s_ood >= t).astype(int))
    results.append((C, float(f_in), float(f_ood)))
    print(C, round(f_in, 4), round(f_ood, 4))

mean_in = float(np.mean([r[1] for r in results]))
mean_ood = float(np.mean([r[2] for r in results]))
print("MEAN_INDIST", round(mean_in, 4))
print("MEAN_OOD", round(mean_ood, 4))
print(json.dumps([{"concept": r[0], "indist_f1": r[1], "ood_f1": r[2]} for r in results]))
