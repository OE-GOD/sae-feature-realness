import torch, json
torch.manual_seed(0)

DEV = "cpu"
PATH = "/Users/oe/rebuild/gemma_detector_dataset.pt"
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 200

d = torch.load(PATH, map_location="cpu")

train_F = d["train_F"].float()
test_F  = d["test_F"].float()
ts_F    = d["ts_F"].float()
wk_F    = d["wk_F"].float()


def f1_at_thresh(scores, y, thr):
    pred = (scores >= thr).int()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def best_thresh(scores, y):
    # candidate thresholds from sorted unique scores
    cand = torch.unique(scores)
    if cand.numel() > 512:
        idx = torch.linspace(0, cand.numel() - 1, 512).long()
        cand = cand[idx]
    best_f1, best_t = 0.0, 0.5
    for t in cand.tolist():
        f = f1_at_thresh(scores, y, t)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def train_mlp(X, y, in_dim, epochs=200, lr=1e-2, wd=1e-4):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    # class weighting for imbalance
    pos = float(y.sum()); neg = float((y == 0).sum())
    pw = torch.tensor([neg / max(pos, 1.0)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    yf = y.float().unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, yf)
        loss.backward()
        opt.step()
    return model


results = {}
for C in CONCEPTS:
    ytr = d["train_L"][C].int()
    if int(ytr.sum()) < 5:
        continue

    # 1. SELECT top-k |point-biserial corr| on train
    Xtr = train_F
    yf = ytr.float()
    # point-biserial == Pearson corr between feature and binary label
    xm = Xtr.mean(0); xs = Xtr.std(0) + 1e-8
    ym = yf.mean(); ysd = yf.std() + 1e-8
    cov = ((Xtr - xm) * (yf.unsqueeze(1) - ym)).mean(0)
    corr = cov / (xs * ysd)
    corr = torch.nan_to_num(corr, nan=0.0)
    sel = torch.topk(corr.abs(), NF).indices

    # 2. Standardize using train mean/std on selected cols
    mu = Xtr[:, sel].mean(0)
    sd = Xtr[:, sel].std(0) + 1e-8
    Xtr_s = (Xtr[:, sel] - mu) / sd

    # 3. Train MLP probe
    model = train_mlp(Xtr_s, ytr, NF)

    # 4. Tune threshold on train
    with torch.no_grad():
        tr_scores = torch.sigmoid(model(Xtr_s)).squeeze(1)
    thr = best_thresh(tr_scores, ytr)

    def eval_split(F, L):
        Xs = (F[:, sel] - mu) / sd
        with torch.no_grad():
            sc = torch.sigmoid(model(Xs)).squeeze(1)
        return f1_at_thresh(sc, L.int(), thr)

    indist = eval_split(test_F, d["test_L"][C])

    rec = {"concept": C, "indist_f1": indist}

    yts = d["ts_L"][C].int()
    if int(yts.sum()) >= 3:
        rec["ts_f1"] = eval_split(ts_F, yts)
    ywk = d["wk_L"][C].int()
    if int(ywk.sum()) >= 3:
        rec["wk_f1"] = eval_split(wk_F, ywk)

    results[C] = rec
    print(C, rec)

per_concept = list(results.values())
mean_indist = sum(r["indist_f1"] for r in per_concept) / len(per_concept)
ood_vals = []
for r in per_concept:
    if "ts_f1" in r:
        ood_vals.append(r["ts_f1"])
    if "wk_f1" in r:
        ood_vals.append(r["wk_f1"])
mean_ood = sum(ood_vals) / len(ood_vals)

out = {
    "recipe": "corr|nf200|mlp",
    "mean_indist_f1": mean_indist,
    "mean_ood_f1": mean_ood,
    "per_concept": per_concept,
}
print(json.dumps(out, indent=2))
with open("/Users/oe/rebuild/result_corr_nf200_mlp.json", "w") as f:
    json.dump(out, f, indent=2)
