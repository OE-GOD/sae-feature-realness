import torch
import numpy as np
import json

D = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = D["keys"]

train_F = D["train_F"].float().numpy()
test_F = D["test_F"].float().numpy()
ood_F = D["ood_F"].float().numpy()

N_FEATURES = 100

def f1_score(y, pred):
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def train_logreg(X, y, epochs=300, lr=0.5, wd=0.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    yf = y.astype(np.float64)
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad = p - yf
        gw = X.T @ grad / n + wd * w
        gb = grad.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b

def predict_proba(X, w, b):
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))

def best_threshold(scores, y):
    thr = np.unique(scores)
    if len(thr) > 500:
        thr = np.quantile(scores, np.linspace(0, 1, 500))
    best_f1, best_t = -1.0, 0.5
    for t in thr:
        pred = (scores >= t).astype(int)
        f = f1_score(y, pred)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

results = []
for C in keys:
    ytr = D["train_L"][C].long().numpy()
    if ytr.sum() < 5:
        continue
    yte = D["test_L"][C].long().numpy()
    yoo = D["ood_L"][C].long().numpy()

    # meandiff selection
    m1 = train_F[ytr == 1].mean(axis=0)
    m0 = train_F[ytr == 0].mean(axis=0)
    diff = np.abs(m1 - m0)
    cols = np.argsort(diff)[::-1][:N_FEATURES]

    Xtr = train_F[:, cols]
    Xte = test_F[:, cols]
    Xoo = ood_F[:, cols]

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xoo = (Xoo - mu) / sd

    w, b = train_logreg(Xtr, ytr)

    str_scores = predict_proba(Xtr, w, b)
    t = best_threshold(str_scores, ytr)

    te_pred = (predict_proba(Xte, w, b) >= t).astype(int)
    oo_pred = (predict_proba(Xoo, w, b) >= t).astype(int)

    f1_te = f1_score(yte, te_pred)
    f1_oo = f1_score(yoo, oo_pred)
    results.append((C, float(f1_te), float(f1_oo)))
    print(f"{C}: indist={f1_te:.4f} ood={f1_oo:.4f}")

mean_in = float(np.mean([r[1] for r in results]))
mean_oo = float(np.mean([r[2] for r in results]))
print(f"\nMEAN indist={mean_in:.4f} ood={mean_oo:.4f}")
print("JSON:" + json.dumps({"mean_indist_f1": mean_in, "mean_ood_f1": mean_oo,
    "per_concept": [{"concept": r[0], "indist_f1": r[1], "ood_f1": r[2]} for r in results]}))
