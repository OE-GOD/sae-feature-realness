import torch, json
torch.manual_seed(0)

PATH = "/Users/oe/rebuild/gemma_detector_dataset.pt"
N_FEATURES = 200

data = torch.load(PATH, map_location="cpu")
concepts = data["keys"] if isinstance(data.get("keys"), list) else list(data["train_L"].keys())

train_F = data["train_F"]
test_F = data["test_F"]
ts_F = data["ts_F"]
wk_F = data["wk_F"]

def logreg_fit(X, y, epochs=400, lr=0.5, wd=0.0):
    n, d = X.shape
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    pos = y.sum().item(); neg = n - pos
    pw = torch.tensor([neg / max(pos, 1)])
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(X @ w + b, y)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()

def proba(X, w, b):
    return torch.sigmoid(X @ w + b)

def f1_from_pred(pred, y):
    tp = (pred & (y == 1)).sum().item()
    fp = (pred & (y == 0)).sum().item()
    fn = ((~pred) & (y == 1)).sum().item()
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else 0.0

def best_threshold(p, y):
    cands = torch.unique(p)
    if cands.numel() > 600:
        cands = torch.quantile(p, torch.linspace(0, 1, 600))
    best_t, best_f1 = 0.5, -1.0
    yb = (y == 1)
    for t in cands:
        pred = p >= t
        f1 = f1_from_pred(pred, y)
        if f1 > best_f1:
            best_f1, best_t = f1, t.item()
    return best_t

results = []
for C in concepts:
    ytr = data["train_L"][C].float()
    if ytr.sum() < 5:
        continue
    Xtr = train_F.float()
    pos_mask = ytr == 1
    neg_mask = ytr == 0
    md = (Xtr[pos_mask].mean(0) - Xtr[neg_mask].mean(0)).abs()
    cols = torch.topk(md, N_FEATURES).indices

    Xtr_s = Xtr[:, cols]
    mu = Xtr_s.mean(0)
    sd = Xtr_s.std(0)
    sd[sd == 0] = 1.0
    Xtr_z = (Xtr_s - mu) / sd

    w, b = logreg_fit(Xtr_z, ytr)
    ptr = proba(Xtr_z, w, b)
    thr = best_threshold(ptr, ytr)

    def eval_split(Fsplit, lab):
        y = data[lab][C].float()
        npos = int(y.sum().item())
        Xz = (Fsplit.float()[:, cols] - mu) / sd
        p = proba(Xz, w, b)
        pred = p >= thr
        return f1_from_pred(pred, y), npos

    indist_f1, _ = eval_split(test_F, "test_L")
    ts_f1, ts_pos = eval_split(ts_F, "ts_L")
    wk_f1, wk_pos = eval_split(wk_F, "wk_L")

    entry = {"concept": C, "indist_f1": round(indist_f1, 4)}
    if ts_pos >= 3:
        entry["ts_f1"] = round(ts_f1, 4)
    if wk_pos >= 3:
        entry["wk_f1"] = round(wk_f1, 4)
    results.append(entry)
    del Xtr, Xtr_s, Xtr_z

mean_indist = sum(r["indist_f1"] for r in results) / len(results)
ood_vals = []
for r in results:
    if "ts_f1" in r:
        ood_vals.append(r["ts_f1"])
    if "wk_f1" in r:
        ood_vals.append(r["wk_f1"])
mean_ood = sum(ood_vals) / len(ood_vals)

out = {"recipe": "meandiff|nf200|linear",
       "mean_indist_f1": round(mean_indist, 4),
       "mean_ood_f1": round(mean_ood, 4),
       "per_concept": results}
print("JSON_START")
print(json.dumps(out, indent=2))
print("JSON_END")
