import torch, math, json
import numpy as np

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float().numpy()
teF = d["test_F"].float().numpy()
ooF = d["ood_F"].float().numpy()

NF = 30

def f1(y, pred):
    y = y.astype(int); pred = pred.astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    if tp == 0: return 0.0
    p = tp/(tp+fp); r = tp/(tp+fn)
    return 2*p*r/(p+r)

def select_mi(F, y, k):
    B = (F > 0).astype(np.float64)
    n = len(y)
    y = y.astype(np.float64)
    py1 = y.mean(); py0 = 1 - py1
    mis = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        b = B[:, j]
        px1 = b.mean(); px0 = 1 - px1
        mi = 0.0
        for bv, px in ((1, px1), (0, px0)):
            for yv, py in ((1, py1), (0, py0)):
                cnt = np.sum((b == bv) & (y == yv))
                if cnt == 0: continue
                pxy = cnt / n
                if px > 0 and py > 0:
                    mi += pxy * math.log(pxy / (px * py))
        mis[j] = mi
    return np.argsort(-mis)[:k]

def train_logreg(X, y, lr=0.1, iters=2000, l2=1e-4):
    n, m = X.shape
    w = np.zeros(m); b = 0.0
    yf = y.astype(float)
    for _ in range(iters):
        z = X@w + b
        p = 1/(1+np.exp(-z))
        g = p - yf
        gw = X.T@g/n + l2*w
        gb = g.mean()
        w -= lr*gw; b -= lr*gb
    return w, b

def proba(X, w, b):
    return 1/(1+np.exp(-(X@w+b)))

def best_thresh(probs, y):
    bestf, bestt = -1, 0.5
    for t in np.unique(probs):
        f = f1(y, (probs >= t).astype(int))
        if f > bestf: bestf, bestt = f, t
    return bestt

rows = []
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5: continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)
    cols = select_mi(trF, ytr, NF)
    Xtr = trF[:, cols]; mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr-mu)/sd; Xte = (teF[:, cols]-mu)/sd; Xoo = (ooF[:, cols]-mu)/sd
    w, b = train_logreg(Xtr, ytr)
    ptr = proba(Xtr, w, b)
    t = best_thresh(ptr, ytr)
    f_te = f1(yte, (proba(Xte, w, b) >= t).astype(int))
    f_oo = f1(yoo, (proba(Xoo, w, b) >= t).astype(int))
    rows.append((C, float(f_te), float(f_oo)))
    print(C, f_te, f_oo)

mte = np.mean([r[1] for r in rows]); moo = np.mean([r[2] for r in rows])
print("MEAN_INDIST", mte); print("MEAN_OOD", moo)
print(json.dumps(rows))
