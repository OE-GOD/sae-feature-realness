"""
best_detectors.py
=================
Builds the BEST detector per concept using the WINNING recipe from the
detector-recipe sweep, ranked by mean cross-distribution (OOD) F1:

    WINNER:  l1 | nf100 | mlp     (mean OOD F1 = 0.7790)

Recipe = feature SELECTION via L1-regularized logistic regression over all
2048 SAE features -> keep top 100 by |weight| -> train a small MLP classifier
on those 100 standardized features -> pick the decision threshold that
maximizes F1 on the TRAIN set -> evaluate on in-dist test and OOD splits.

For each concept it prints in-dist + OOD F1 and saves every trained detector
(MLP state, selected feature columns, normalization stats, threshold) to
best_detectors.pt.

This is the same procedure used by eval_l1_mlp.py (the sweep's winning recipe);
it is reproduced here as the canonical "build the best detectors" script.
"""

import torch, numpy as np, json

torch.manual_seed(0)
np.random.seed(0)

DATA = "/Users/oe/rebuild/detector_dataset.pt"
OUT = "/Users/oe/rebuild/best_detectors.pt"
RECIPE = "l1|nf100|mlp"
NF = 100  # number of features kept by the selector

d = torch.load(DATA, weights_only=False)
keys = d["keys"]
trF = d["train_F"].float()
teF = d["test_F"].float()
ooF = d["ood_F"].float()


def f1(y, pred):
    y = y.astype(int); pred = pred.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp); r = tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def best_thresh(probs, y):
    ts = np.unique(probs)
    if len(ts) > 500:
        ts = np.quantile(probs, np.linspace(0, 1, 500))
    bestf, bestt = -1.0, 0.5
    for t in ts:
        f = f1(y, (probs >= t).astype(int))
        if f > bestf:
            bestf, bestt = f, t
    return bestt


def l1_logreg_select(X, y, k, l1=1e-3, lr=0.05, iters=1500):
    """L1-regularized logistic regression over ALL features; return top-k by |w|."""
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
        loss.backward()
        opt.step()
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
    m = X.shape[1]
    model = MLP(m)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()
    yt = y.float()
    for _ in range(iters):
        opt.zero_grad()
        z = model(X)
        loss = bce(z, yt)
        loss.backward()
        opt.step()
    return model


print(f"WINNING RECIPE: {RECIPE}  (NF={NF})\n")
print(f"{'concept':<12} {'in-dist F1':>11} {'OOD F1':>9}")
print("-" * 34)

detectors = {}
rows = []
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)
    ytr_t = torch.from_numpy(ytr)

    # SELECTION: L1 logreg on ALL 2048 features (standardized full set).
    fmu = trF.mean(0); fsd = trF.std(0) + 1e-8
    Xfull = (trF - fmu) / fsd
    cols = l1_logreg_select(Xfull, ytr_t, NF)

    # Standardize the selected features using TRAIN mean/std.
    Xtr = trF[:, cols]; mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (teF[:, cols] - mu) / sd
    Xoo = (ooF[:, cols] - mu) / sd

    model = train_mlp(Xtr, ytr_t)
    with torch.no_grad():
        ptr = torch.sigmoid(model(Xtr)).numpy()
        pte = torch.sigmoid(model(Xte)).numpy()
        poo = torch.sigmoid(model(Xoo)).numpy()
    t = best_thresh(ptr, ytr)
    f_te = f1(yte, (pte >= t).astype(int))
    f_oo = f1(yoo, (poo >= t).astype(int))
    rows.append((C, float(f_te), float(f_oo)))
    print(f"{C:<12} {f_te:>11.4f} {f_oo:>9.4f}")

    detectors[C] = {
        "cols": cols.cpu(),          # selected feature indices into the 2048-d SAE space
        "mu": mu.cpu(),              # per-feature train mean (for standardization)
        "sd": sd.cpu(),             # per-feature train std
        "threshold": float(t),       # F1-optimal decision threshold (chosen on train)
        "mlp_state": model.state_dict(),
        "indist_f1": float(f_te),
        "ood_f1": float(f_oo),
    }

mte = float(np.mean([r[1] for r in rows]))
moo = float(np.mean([r[2] for r in rows]))
print("-" * 34)
print(f"{'MEAN':<12} {mte:>11.4f} {moo:>9.4f}")

torch.save({
    "recipe": RECIPE,
    "nf": NF,
    "keys": [r[0] for r in rows],
    "detectors": detectors,
    "mean_indist_f1": mte,
    "mean_ood_f1": moo,
}, OUT)
print(f"\nSaved {len(detectors)} detectors -> {OUT}")
print("RESULT_JSON", json.dumps({
    "recipe": RECIPE,
    "mean_indist_f1": mte, "mean_ood_f1": moo,
    "per_concept": [{"concept": r[0], "indist_f1": r[1], "ood_f1": r[2]} for r in rows]
}))
