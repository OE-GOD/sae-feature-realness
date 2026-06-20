import torch
import numpy as np
import json

torch.manual_seed(0)
np.random.seed(0)

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float()
test_F = d["test_F"].float()
ood_F = d["ood_F"].float()

N_FEAT = 30


def fit_logreg(X, y, l1=0.0, l2=0.0, epochs=300, lr=0.5):
    n, dim = X.shape
    w = torch.zeros(dim, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    yt = y.float()
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = X @ w + b
        loss = bce(logits, yt)
        if l2 > 0:
            loss = loss + l2 * (w * w).sum()
        if l1 > 0:
            loss = loss + l1 * w.abs().sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def best_f1_threshold(scores, y):
    cands = torch.unique(scores)
    if cands.numel() > 1:
        mids = (cands[:-1] + cands[1:]) / 2.0
        thr_cands = torch.cat([torch.tensor([-1e9]), mids, torch.tensor([1e9])])
    else:
        thr_cands = torch.tensor([-1e9, 1e9])
    P = y.sum().item()
    best_t, best_f = 0.0, -1.0
    for t in thr_cands.tolist():
        pred = scores >= t
        tp = (pred & (y == 1)).sum().item()
        fp = (pred & (y == 0)).sum().item()
        fn = P - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        if f1 > best_f:
            best_f, best_t = f1, t
    return best_t


def f1_at(scores, y, thr):
    pred = scores >= thr
    tp = (pred & (y == 1)).sum().item()
    fp = (pred & (y == 0)).sum().item()
    fn = ((~pred) & (y == 1)).sum().item()
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 0.0


results = []
for C in keys:
    ytr = d["train_L"][C].long()
    if ytr.sum().item() < 5:
        continue
    yte = d["test_L"][C].long()
    yood = d["ood_L"][C].long()

    # SELECTION: L1 logistic regression on ALL 2048 features (standardized for stable fit),
    # take top-k by |weight|. Standardize all-feature matrix for selection fit.
    mu_all = train_F.mean(0)
    sd_all = train_F.std(0)
    sd_all[sd_all == 0] = 1.0
    Xall = (train_F - mu_all) / sd_all
    w_all, _ = fit_logreg(Xall, ytr, l1=1e-3, epochs=400, lr=0.5)
    cols = torch.argsort(w_all.abs(), descending=True)[:N_FEAT]

    Xtr = train_F[:, cols]
    Xte = test_F[:, cols]
    Xood = ood_F[:, cols]

    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    Xood_s = (Xood - mu) / sd

    # PROBE: linear logistic regression
    w, b = fit_logreg(Xtr_s, ytr, l2=1e-4, epochs=400, lr=0.1)

    s_tr = Xtr_s @ w + b
    thr = best_f1_threshold(s_tr, ytr)

    s_te = Xte_s @ w + b
    s_ood = Xood_s @ w + b

    indist_f1 = f1_at(s_te, yte, thr)
    ood_f1 = f1_at(s_ood, yood, thr)
    results.append((C, float(indist_f1), float(ood_f1)))
    print(f"{C}: indist={indist_f1:.4f} ood={ood_f1:.4f}")

mean_in = float(np.mean([r[1] for r in results]))
mean_ood = float(np.mean([r[2] for r in results]))
print(f"\nMEAN indist={mean_in:.4f} ood={mean_ood:.4f}")
print("JSON", json.dumps({
    "mean_indist_f1": mean_in, "mean_ood_f1": mean_ood,
    "per_concept": [{"concept": r[0], "indist_f1": r[1], "ood_f1": r[2]} for r in results]
}))
