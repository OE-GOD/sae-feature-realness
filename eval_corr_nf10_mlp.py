import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float()
test_F = d["test_F"].float()
ood_F = d["ood_F"].float()

SELECTION = "corr"
N_FEATURES = 10
PROBE = "mlp"


def select_corr(F, y, k):
    y = y.float()
    ym = y.mean()
    ys = y.std(unbiased=False) + 1e-8
    Fm = F.mean(0)
    Fs = F.std(0, unbiased=False) + 1e-8
    # point-biserial = pearson with binary label
    cov = ((F - Fm) * (y - ym).unsqueeze(1)).mean(0)
    corr = cov / (Fs * ys)
    corr = torch.nan_to_num(corr, nan=0.0)
    idx = torch.argsort(corr.abs(), descending=True)[:k]
    return idx


def f1_score(y, pred):
    y = y.bool()
    pred = pred.bool()
    tp = (y & pred).sum().item()
    fp = (~y & pred).sum().item()
    fn = (y & ~pred).sum().item()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def best_threshold(scores, y):
    # tune threshold on train to max F1
    s = scores.numpy()
    order = np.unique(s)
    cands = order
    if len(cands) > 200:
        qs = np.quantile(s, np.linspace(0, 1, 200))
        cands = np.unique(qs)
    best_f1, best_t = -1, 0.5
    for t in cands:
        pred = torch.from_numpy((s > t).astype(np.float32))
        f = f1_score(y, pred)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


class MLP(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y):
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    yf = y.float()
    for _ in range(300):
        opt.zero_grad()
        out = model(X)
        loss = lossf(out, yf)
        loss.backward()
        opt.step()
    return model


results = []
indist_f1s = []
ood_f1s = []

for C in keys:
    ytr = d["train_L"][C].float()
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].float()
    yood = d["ood_L"][C].float()

    idx = select_corr(train_F, ytr, N_FEATURES)

    Xtr = train_F[:, idx]
    Xte = test_F[:, idx]
    Xood = ood_F[:, idx]

    mu = Xtr.mean(0)
    sd = Xtr.std(0, unbiased=False) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    Xood_s = (Xood - mu) / sd

    model = train_mlp(Xtr_s, ytr)
    model.eval()
    with torch.no_grad():
        str_scores = torch.sigmoid(model(Xtr_s))
        ste_scores = torch.sigmoid(model(Xte_s))
        sood_scores = torch.sigmoid(model(Xood_s))

    t = best_threshold(str_scores, ytr)

    f1_te = f1_score(yte, (ste_scores > t).float())
    f1_ood = f1_score(yood, (sood_scores > t).float())

    results.append((C, f1_te, f1_ood))
    indist_f1s.append(f1_te)
    ood_f1s.append(f1_ood)
    print(f"{C}: indist={f1_te:.4f} ood={f1_ood:.4f}")

mean_in = float(np.mean(indist_f1s))
mean_ood = float(np.mean(ood_f1s))
print(f"\nMEAN indist={mean_in:.4f} ood={mean_ood:.4f}")
import json
print("JSON_OUT", json.dumps({
    "mean_indist_f1": mean_in,
    "mean_ood_f1": mean_ood,
    "per_concept": [{"concept": c, "indist_f1": a, "ood_f1": b} for c, a, b in results],
}))
