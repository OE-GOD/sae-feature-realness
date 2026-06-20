import torch
import numpy as np

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float().numpy()
test_F = d["test_F"].float().numpy()
ood_F = d["ood_F"].float().numpy()

n_features = 10

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

def best_threshold(y, scores):
    cand = np.unique(scores)
    if len(cand) > 1:
        mids = (cand[:-1] + cand[1:]) / 2
    else:
        mids = cand
    ts = np.concatenate([[scores.min() - 1e-6], mids, [scores.max() + 1e-6]])
    best_t, best_f = ts[0], -1.0
    for t in ts:
        pred = (scores >= t).astype(int)
        f = f1_score(y, pred)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

def train_logreg(X, y, epochs=500, lr=0.1, wd=0.0):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n, dim = Xt.shape
    w = torch.zeros(dim, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = Xt @ w + b
        loss = lossf(logits, yt)
        loss.backward()
        opt.step()
    return w.detach().numpy(), b.detach().numpy()

def decision(X, w, b):
    return X @ w + b

results = []
indist_list = []
ood_list = []

for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].numpy().astype(int)
    yood = d["ood_L"][C].numpy().astype(int)

    mu1 = train_F[ytr == 1].mean(axis=0)
    mu0 = train_F[ytr == 0].mean(axis=0)
    score = np.abs(mu1 - mu0)
    cols = np.argsort(score)[::-1][:n_features]

    Xtr = train_F[:, cols]
    Xte = test_F[:, cols]
    Xood = ood_F[:, cols]

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    Xood = (Xood - mean) / std

    w, b = train_logreg(Xtr, ytr)

    str_tr = decision(Xtr, w, b)
    t = best_threshold(ytr, str_tr)

    pred_te = (decision(Xte, w, b) >= t).astype(int)
    pred_ood = (decision(Xood, w, b) >= t).astype(int)

    f_te = f1_score(yte, pred_te)
    f_ood = f1_score(yood, pred_ood)

    results.append((C, f_te, f_ood))
    indist_list.append(f_te)
    ood_list.append(f_ood)

print("RESULTS")
for C, f_te, f_ood in results:
    print(f"{C}\t{f_te:.6f}\t{f_ood:.6f}")
print(f"MEAN_INDIST\t{np.mean(indist_list):.6f}")
print(f"MEAN_OOD\t{np.mean(ood_list):.6f}")
