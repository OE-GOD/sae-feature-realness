import torch
import torch.nn as nn

torch.manual_seed(0)
device = "cpu"

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float()
test_F = d["test_F"].float()
ood_F = d["ood_F"].float()
train_L = d["train_L"]
test_L = d["test_L"]
ood_L = d["ood_L"]

N_FEATURES = 30


def f1_score(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def train_logreg(X, y, l1=0.0, epochs=300, lr=0.1, wd=0.0):
    n, dim = X.shape
    w = torch.zeros(dim, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    lossfn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = X @ w + b
        loss = lossfn(logits, y)
        if l1 > 0:
            loss = loss + l1 * w.abs().sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


class MLP(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y, epochs=300, lr=0.01, wd=1e-4):
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossfn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = lossfn(logits, y)
        loss.backward()
        opt.step()
    return model


def best_threshold(scores, y):
    # scores: 1D tensor of probabilities/logits; pick threshold maximizing F1 on train
    uniq = torch.unique(scores)
    if len(uniq) > 200:
        qs = torch.linspace(0, 1, 200)
        cands = torch.quantile(scores, qs)
    else:
        cands = uniq
    best_t, best_f = 0.0, -1.0
    for t in cands:
        pred = (scores >= t).long()
        f = f1_score(y, pred)
        if f > best_f:
            best_f, best_t = f, t.item()
    return best_t


results = []
for C in keys:
    ytr = train_L[C].float()
    if ytr.sum() < 5:
        continue

    # SELECTION: l1 -> L1 logreg on ALL 2048 features, top-k by |weight|
    mean_all = train_F.mean(0)
    std_all = train_F.std(0) + 1e-8
    Xall = (train_F - mean_all) / std_all
    w, b = train_logreg(Xall, ytr, l1=1e-3, epochs=300, lr=0.05)
    cols = torch.topk(w.abs(), N_FEATURES).indices

    # Standardize selected features using train mean/std
    tr_sel = train_F[:, cols]
    te_sel = test_F[:, cols]
    ood_sel = ood_F[:, cols]
    mu = tr_sel.mean(0)
    sd = tr_sel.std(0) + 1e-8
    Xtr = (tr_sel - mu) / sd
    Xte = (te_sel - mu) / sd
    Xood = (ood_sel - mu) / sd

    # PROBE: mlp
    model = train_mlp(Xtr, ytr)
    with torch.no_grad():
        s_tr = torch.sigmoid(model(Xtr))
        s_te = torch.sigmoid(model(Xte))
        s_ood = torch.sigmoid(model(Xood))

    t = best_threshold(s_tr, train_L[C].long())

    f1_te = f1_score(test_L[C].long(), (s_te >= t).long())
    f1_ood = f1_score(ood_L[C].long(), (s_ood >= t).long())

    results.append((C, f1_te, f1_ood))
    print(f"{C}: indist={f1_te:.4f} ood={f1_ood:.4f}")

mean_ind = sum(r[1] for r in results) / len(results)
mean_ood = sum(r[2] for r in results) / len(results)
print(f"\nMEAN indist={mean_ind:.4f} ood={mean_ood:.4f}")
import json
print("JSON:", json.dumps({
    "mean_indist_f1": mean_ind,
    "mean_ood_f1": mean_ood,
    "per_concept": [{"concept": c, "indist_f1": a, "ood_f1": o} for c, a, o in results],
}))
