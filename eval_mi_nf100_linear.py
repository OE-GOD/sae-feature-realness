import torch
import numpy as np

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float().numpy()
test_F = d["test_F"].float().numpy()
ood_F = d["ood_F"].float().numpy()

n_features = 100

def mi_select(F, y, k):
    B = (F > 0).astype(int)
    y = y.astype(int)
    n = len(y)
    mis = np.zeros(F.shape[1])
    py1 = y.mean(); py0 = 1 - py1
    for j in range(F.shape[1]):
        b = B[:, j]
        mi = 0.0
        for bv, py, pyv in ((0, py0, 0), (1, py1, 1)):
            pass
        for bv in (0, 1):
            mask_b = (b == bv)
            pxb = mask_b.mean()
            if pxb == 0:
                continue
            for yv, pyv in ((0, py0), (1, py1)):
                n11 = np.sum(mask_b & (y == yv))
                if n11 == 0:
                    continue
                pxy = n11 / n
                mi += pxy * np.log(pxy / (pxb * pyv))
        mis[j] = mi
    return np.argsort(mis)[::-1][:k]

def f1_score(y, pred):
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)

def train_logreg(X, y, epochs=300, lr=0.5):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = Xt @ w + b
        loss = lossf(logits, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return w.detach(), b.detach()

def predict_proba(X, w, b):
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(Xt @ w + b).numpy()

def best_threshold(probs, y):
    best_f1, best_t = -1, 0.5
    for t in np.unique(probs):
        f1 = f1_score(y, (probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

results = []
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].numpy().astype(int)
    yood = d["ood_L"][C].numpy().astype(int)

    cols = mi_select(train_F, ytr, n_features)

    Xtr = train_F[:, cols]
    Xte = test_F[:, cols]
    Xood = ood_F[:, cols]

    mean = Xtr.mean(0); std = Xtr.std(0); std[std == 0] = 1.0
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    Xood = (Xood - mean) / std

    w, b = train_logreg(Xtr, ytr)

    ptr = predict_proba(Xtr, w, b)
    t = best_threshold(ptr, ytr)

    pte = predict_proba(Xte, w, b)
    pood = predict_proba(Xood, w, b)

    f1_te = f1_score(yte, (pte >= t).astype(int))
    f1_ood = f1_score(yood, (pood >= t).astype(int))
    results.append((C, float(f1_te), float(f1_ood)))
    print(f"{C}: indist={f1_te:.4f} ood={f1_ood:.4f}")

mean_in = float(np.mean([r[1] for r in results]))
mean_ood = float(np.mean([r[2] for r in results]))
print(f"MEAN indist={mean_in:.4f} ood={mean_ood:.4f}")
import json
print(json.dumps({"mean_indist": mean_in, "mean_ood": mean_ood, "per": results}))
