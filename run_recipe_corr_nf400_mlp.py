import torch, gc, json

torch.manual_seed(0)
PATH = "/Users/oe/rebuild/gemma_detector_dataset.pt"
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 400
EPS = 1e-6

data = torch.load(PATH, map_location="cpu")

def f1_at(scores, y):
    order = torch.argsort(scores, descending=True)
    ys = y[order].float()
    P = ys.sum().item()
    if P == 0:
        return 0.0, scores.max().item() + 1.0
    tp = torch.cumsum(ys, dim=0)
    fp = torch.cumsum(1 - ys, dim=0)
    prec = tp / (tp + fp)
    rec = tp / P
    f1 = 2 * prec * rec / (prec + rec + EPS)
    best = torch.argmax(f1).item()
    s_sorted = scores[order]
    thr = s_sorted[best].item() - 1e-9
    return f1[best].item(), thr

def f1_with_thr(scores, y, thr):
    pred = (scores >= thr).float()
    tp = (pred * y).sum().item()
    fp = (pred * (1 - y)).sum().item()
    fn = ((1 - pred) * y).sum().item()
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec + EPS)

class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

results = {}
N = data["train_F"].shape[0]

for C in CONCEPTS:
    ytr = data["train_L"][C].float()
    if ytr.sum().item() < 5:
        print(f"skip {C}: <5 positives")
        continue

    X = data["train_F"].float()
    y = ytr
    yc = y - y.mean()
    Xc = X - X.mean(0, keepdim=True)
    num = (Xc * yc.unsqueeze(1)).sum(0)
    denom = Xc.pow(2).sum(0).sqrt() * yc.pow(2).sum().sqrt()
    corr = (num / (denom + EPS)).abs()
    cols = torch.topk(corr, NF).indices

    Xs = X[:, cols]
    mean = Xs.mean(0, keepdim=True)
    std = Xs.std(0, keepdim=True) + EPS
    Xtr = (Xs - mean) / std

    del X, Xc, corr, num, denom
    gc.collect()

    torch.manual_seed(0)
    model = MLP(NF)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = y.sum().item()
    neg = N - pos
    pw = torch.tensor(neg / max(pos, 1.0))
    lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    model.train()
    for ep in range(200):
        opt.zero_grad()
        loss = lossfn(model(Xtr), y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_scores = model(Xtr)
    _, thr = f1_at(tr_scores, y)

    def eval_split(Fkey, Lkey):
        F = data[Fkey][:, cols].float()
        Fs = (F - mean) / std
        yy = data[Lkey][C].float()
        with torch.no_grad():
            sc = model(Fs)
        return sc, yy

    sc, yy = eval_split("test_F", "test_L")
    indist = f1_with_thr(sc, yy, thr)

    ts_f1 = None
    sc, yy = eval_split("ts_F", "ts_L")
    if yy.sum().item() >= 3:
        ts_f1 = f1_with_thr(sc, yy, thr)

    wk_f1 = None
    sc, yy = eval_split("wk_F", "wk_L")
    if yy.sum().item() >= 3:
        wk_f1 = f1_with_thr(sc, yy, thr)

    results[C] = dict(indist=indist, ts=ts_f1, wk=wk_f1)
    print(f"{C}: indist={indist:.4f} ts={ts_f1} wk={wk_f1}")
    del Xs, Xtr, model
    gc.collect()

inds = [r["indist"] for r in results.values()]
oods = []
for r in results.values():
    if r["ts"] is not None:
        oods.append(r["ts"])
    if r["wk"] is not None:
        oods.append(r["wk"])
mean_ind = sum(inds)/len(inds)
mean_ood = sum(oods)/len(oods)
print("MEAN_INDIST", mean_ind)
print("MEAN_OOD", mean_ood)
print("JSON", json.dumps({"results": results, "mean_indist": mean_ind, "mean_ood": mean_ood}))
