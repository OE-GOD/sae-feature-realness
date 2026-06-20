import torch, json

torch.manual_seed(0)
d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float()
test_F = d["test_F"].float()
ood_F = d["ood_F"].float()

N_FEATURES = 10
SELECTION = "l1"
PROBE = "mlp"


def logreg(X, y, l1=0.0, l2=0.0, epochs=300, lr=0.1):
    dim = X.shape[1]
    w = torch.zeros(dim, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(X @ w + b, y)
        if l1 > 0:
            loss = loss + l1 * w.abs().sum()
        if l2 > 0:
            loss = loss + l2 * (w**2).sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


class MLP(torch.nn.Module):
    def __init__(self, dim, h=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, h), torch.nn.ReLU(), torch.nn.Linear(h, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y, epochs=300, lr=0.01, wd=1e-4):
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = bce(m(X), y)
        loss.backward()
        opt.step()
    return m


def f1_at(scores, y, thr):
    pred = (scores >= thr).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def best_thr(scores, y):
    cand = torch.unique(scores)
    if len(cand) > 300:
        cand = torch.quantile(scores, torch.linspace(0, 1, 300))
    best_f, best_t = 0.0, cand.min().item() - 1e-6
    for t in cand:
        f = f1_at(scores, y, t.item())
        if f > best_f:
            best_f, best_t = f, t.item()
    return best_t


def select_l1(Xtr, ytr, k):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    Xs = (Xtr - mu) / sd
    w, _ = logreg(Xs, ytr, l1=0.01, epochs=300, lr=0.05)
    return torch.argsort(w.abs(), descending=True)[:k]


per = []
ind_f1s, ood_f1s = [], []
for C in keys:
    ytr = d["train_L"][C].float()
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].float()
    yood = d["ood_L"][C].float()

    cols = select_l1(train_F, ytr, N_FEATURES)

    Xtr = train_F[:, cols]
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (test_F[:, cols] - mu) / sd
    Xood_s = (ood_F[:, cols] - mu) / sd

    m = train_mlp(Xtr_s, ytr)
    with torch.no_grad():
        s_tr, s_te, s_ood = m(Xtr_s), m(Xte_s), m(Xood_s)

    thr = best_thr(s_tr, ytr)
    f_te = f1_at(s_te, yte, thr)
    f_ood = f1_at(s_ood, yood, thr)
    per.append((C, f_te, f_ood))
    ind_f1s.append(f_te)
    ood_f1s.append(f_ood)
    print(f"{C}: indist={f_te:.4f} ood={f_ood:.4f}")

mean_ind = sum(ind_f1s) / len(ind_f1s)
mean_ood = sum(ood_f1s) / len(ood_f1s)
print(f"MEAN indist={mean_ind:.4f} ood={mean_ood:.4f}")
print("JSON " + json.dumps({"mean_indist_f1": mean_ind, "mean_ood_f1": mean_ood,
      "per_concept": [{"concept": c, "indist_f1": a, "ood_f1": b} for c, a, b in per]}))
