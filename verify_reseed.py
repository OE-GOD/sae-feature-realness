"""
Adversarial verification of the winning detector recipe (l1|nf100|mlp).
Re-runs best_detectors.py logic with a DIFFERENT seed for torch/np init
(affects MLP weight init + L1-logreg init/optimization path), and reports
precision AND recall per concept on OOD, plus an always-positive baseline.
"""
import sys, torch, numpy as np, json

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA = "/Users/oe/rebuild/detector_dataset.pt"
NF = 100

d = torch.load(DATA, weights_only=False)
keys = d["keys"]
trF = d["train_F"].float(); teF = d["test_F"].float(); ooF = d["ood_F"].float()


def prf(y, pred):
    y = y.astype(int); pred = pred.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f, tp, fp, fn


def best_thresh(probs, y):
    ts = np.unique(probs)
    if len(ts) > 500:
        ts = np.quantile(probs, np.linspace(0, 1, 500))
    bestf, bestt = -1.0, 0.5
    for t in ts:
        _, _, f, _, _, _ = prf(y, (probs >= t).astype(int))
        if f > bestf:
            bestf, bestt = f, t
    return bestt


def l1_logreg_select(X, y, k, l1=1e-3, lr=0.05, iters=1500):
    n, m = X.shape
    w = torch.zeros(m, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    yt = y.float()
    opt = torch.optim.Adam([w, b], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(iters):
        opt.zero_grad()
        z = X @ w + b
        loss = bce(z, yt) + l1 * w.abs().sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        order = torch.argsort(w.abs(), descending=True)[:k]
    return order


class MLP(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(m, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y, iters=800, lr=1e-2, wd=1e-4):
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss(); yt = y.float()
    for _ in range(iters):
        opt.zero_grad(); loss = bce(model(X), yt); loss.backward(); opt.step()
    return model


print(f"=== RESEED VERIFICATION  seed={SEED}  recipe=l1|nf100|mlp ===\n")
hdr = f"{'concept':<11}{'oodbase':>8}{'OOD_P':>8}{'OOD_R':>8}{'OOD_F1':>8}{'predpos%':>9}{'AP_F1':>7}"
print(hdr); print("-" * len(hdr))

rows = []
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)
    ytr_t = torch.from_numpy(ytr)

    fmu = trF.mean(0); fsd = trF.std(0) + 1e-8
    cols = l1_logreg_select((trF - fmu) / fsd, ytr_t, NF)

    Xtr = trF[:, cols]; mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xoo = (ooF[:, cols] - mu) / sd
    Xte = (teF[:, cols] - mu) / sd

    model = train_mlp(Xtr, ytr_t)
    with torch.no_grad():
        ptr = torch.sigmoid(model(Xtr)).numpy()
        poo = torch.sigmoid(model(Xoo)).numpy()
        pte = torch.sigmoid(model(Xte)).numpy()
    t = best_thresh(ptr, ytr)
    pred_oo = (poo >= t).astype(int)
    p, r, f, tp, fp, fn = prf(yoo, pred_oo)
    _, _, fte, _, _, _ = prf(yte, (pte >= t).astype(int))
    predpos = pred_oo.mean()
    # always-positive baseline F1 on OOD
    _, _, apf, _, _, _ = prf(yoo, np.ones_like(yoo))
    oodbase = yoo.mean()
    rows.append(dict(concept=C, ood_p=p, ood_r=r, ood_f1=f, indist_f1=fte,
                     predpos=float(predpos), ap_f1=apf, oodbase=float(oodbase),
                     tp=tp, fp=fp, fn=fn, ood_pos=int(yoo.sum())))
    print(f"{C:<11}{oodbase:>8.3f}{p:>8.3f}{r:>8.3f}{f:>8.3f}{predpos*100:>8.1f}%{apf:>7.3f}")

mean_f = float(np.mean([x['ood_f1'] for x in rows]))
mean_te = float(np.mean([x['indist_f1'] for x in rows]))
mean_ap = float(np.mean([x['ap_f1'] for x in rows]))
print("-" * len(hdr))
print(f"{'MEAN':<11}{'':>8}{'':>8}{'':>8}{mean_f:>8.3f}{'':>9}{mean_ap:>7.3f}")
print(f"\nmean OOD F1 = {mean_f:.4f}   mean in-dist F1 = {mean_te:.4f}")
print(f"always-positive mean OOD F1 baseline = {mean_ap:.4f}")
print("RESULT_JSON", json.dumps({"seed": SEED, "mean_ood_f1": mean_f,
      "mean_indist_f1": mean_te, "always_pos_mean_ood_f1": mean_ap, "rows": rows}))
