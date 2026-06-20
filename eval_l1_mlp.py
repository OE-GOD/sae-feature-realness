import torch, numpy as np, json

torch.manual_seed(0); np.random.seed(0)
d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float()
teF = d["test_F"].float()
ooF = d["ood_F"].float()

NF = 100
dev = "cpu"

def f1(y, pred):
    y=y.astype(int); pred=pred.astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    if tp==0: return 0.0
    p=tp/(tp+fp); r=tp/(tp+fn)
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def best_thresh(probs, y):
    ts = np.unique(probs)
    if len(ts) > 500:
        ts = np.quantile(probs, np.linspace(0,1,500))
    bestf, bestt = -1.0, 0.5
    for t in ts:
        f = f1(y, (probs>=t).astype(int))
        if f>bestf: bestf,bestt = f,t
    return bestt

def l1_logreg_select(X, y, k, l1=1e-3, lr=0.05, iters=1500):
    # X: torch tensor [n, 2048], standardized full set for fair weight comparison
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
    def forward(self, x): return self.net(x).squeeze(-1)

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

rows = []
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)
    ytr_t = torch.from_numpy(ytr)

    # SELECTION: L1 logreg on ALL 2048 features. Standardize full set for the selection fit.
    fmu = trF.mean(0); fsd = trF.std(0) + 1e-8
    Xfull = (trF - fmu) / fsd
    cols = l1_logreg_select(Xfull, ytr_t, NF)

    # Standardize selected features using train mean/std
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
    print(C, round(f_te,4), round(f_oo,4), flush=True)

mte = float(np.mean([r[1] for r in rows]))
moo = float(np.mean([r[2] for r in rows]))
print("MEAN_INDIST", round(mte,4))
print("MEAN_OOD", round(moo,4))
print("RESULT_JSON", json.dumps({
    "mean_indist_f1": mte, "mean_ood_f1": moo,
    "per_concept": [{"concept":r[0],"indist_f1":r[1],"ood_f1":r[2]} for r in rows]
}))
