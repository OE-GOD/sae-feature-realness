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

n_features = 10

def f1_score(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)

def train_logreg(X, y, l1=0.0, l2=0.0, epochs=300, lr=0.5):
    # X: [N, D] tensor, y: [N] tensor of 0/1
    N, D = X.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    yf = y.float()
    for _ in range(epochs):
        opt.zero_grad()
        logits = X @ w + b
        loss = bce(logits, yf)
        if l1 > 0:
            loss = loss + l1 * w.abs().sum()
        if l2 > 0:
            loss = loss + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()

def best_threshold_f1(probs, y):
    grid = torch.linspace(0, 1, 501)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (probs >= t).int()
        f = f1_score(y, pred)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t

results = []

for C in keys:
    ytr = d["train_L"][C].int()
    if int(ytr.sum()) < 5:
        continue
    yte = d["test_L"][C].int()
    yood = d["ood_L"][C].int()

    # SELECTION: l1 logistic regression on ALL 2048 features, top-k by |weight|
    # standardize all features for the l1 selector
    mu_all = train_F.mean(0)
    sd_all = train_F.std(0)
    sd_all[sd_all == 0] = 1.0
    Xall = (train_F - mu_all) / sd_all
    w_l1, _ = train_logreg(Xall, ytr, l1=1e-3, epochs=300, lr=0.1)
    cols = torch.argsort(w_l1.abs(), descending=True)[:n_features]

    Xtr = train_F[:, cols]
    Xte = test_F[:, cols]
    Xood = ood_F[:, cols]

    # Standardize selected features using train mean/std
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xood = (Xood - mu) / sd

    # PROBE: linear logistic regression
    w, b = train_logreg(Xtr, ytr, l2=1e-4, epochs=400, lr=0.1)

    ptr = torch.sigmoid(Xtr @ w + b)
    t = best_threshold_f1(ptr, ytr)

    pte = torch.sigmoid(Xte @ w + b)
    pood = torch.sigmoid(Xood @ w + b)

    f1_te = f1_score(yte, (pte >= t).int())
    f1_ood = f1_score(yood, (pood >= t).int())

    results.append((C, f1_te, f1_ood))
    print(f"{C}: indist={f1_te:.4f} ood={f1_ood:.4f}")

mean_in = float(np.mean([r[1] for r in results]))
mean_ood = float(np.mean([r[2] for r in results]))
print(f"\nMEAN indist={mean_in:.4f} ood={mean_ood:.4f}")
print("RESULTS_JSON_START")
print(json.dumps({
    "mean_indist_f1": mean_in,
    "mean_ood_f1": mean_ood,
    "per_concept": [{"concept": c, "indist_f1": a, "ood_f1": b} for c, a, b in results]
}))
print("RESULTS_JSON_END")
