import torch, gc, json

torch.manual_seed(0)
PATH = "/Users/oe/rebuild/gemma_detector_dataset.pt"
NF = 100
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]

d = torch.load(PATH, map_location="cpu")
train_F = d["train_F"]  # [N,16384] float16

def f1_at_threshold(scores, y, thr):
    pred = (scores >= thr).float()
    tp = (pred * y).sum()
    fp = (pred * (1 - y)).sum()
    fn = ((1 - pred) * y).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return torch.tensor(0.0)
    return 2 * tp / denom

def best_f1(scores, y):
    s = scores.detach()
    cands = torch.unique(s)
    if cands.numel() > 512:
        idx = torch.linspace(0, cands.numel() - 1, 512).long()
        cands = cands[idx]
    best = torch.tensor(-1.0); best_thr = torch.tensor(0.0)
    for thr in cands:
        f = f1_at_threshold(s, y, thr)
        if f > best:
            best = f; best_thr = thr
    return best, best_thr

results = []
mean_indist = []
ood_vals = []

for C in CONCEPTS:
    ytr = d["train_L"][C].float()
    if ytr.sum() < 5:
        continue

    Xtr_full = train_F.float()  # [N,16384]

    # 1. SELECT meandiff top-NF
    pos_mean = Xtr_full[ytr == 1].mean(0)
    neg_mean = Xtr_full[ytr == 0].mean(0)
    score = (pos_mean - neg_mean).abs()
    cols = torch.topk(score, NF).indices

    Xtr = Xtr_full[:, cols].clone()
    del Xtr_full; gc.collect()

    # 2. Standardize on train
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    Xtr_s = (Xtr - mu) / sd

    # 3. MLP probe: 1 hidden layer 32 ReLU + light weight decay
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(NF, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    npos = ytr.sum(); nneg = (ytr == 0).sum()
    pos_weight = (nneg / npos.clamp(min=1)).clamp(max=50.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    ytr_col = ytr.unsqueeze(1)
    for epoch in range(300):
        opt.zero_grad()
        logits = model(Xtr_s)
        loss = loss_fn(logits, ytr_col)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_scores = model(Xtr_s).squeeze(1)
    _, thr = best_f1(tr_scores, ytr)

    def eval_split(Fkey, Lkey):
        XF = d[Fkey].float()[:, cols]
        y = d[Lkey][C].float()
        XF_s = (XF - mu) / sd
        with torch.no_grad():
            sc = model(XF_s).squeeze(1)
        f = f1_at_threshold(sc, y, thr).item()
        npos_eval = int(y.sum().item())
        del XF, XF_s; gc.collect()
        return f, npos_eval

    indist_f1, _ = eval_split("test_F", "test_L")
    ts_f1, ts_pos = eval_split("ts_F", "ts_L")
    wk_f1, wk_pos = eval_split("wk_F", "wk_L")

    entry = {"concept": C, "indist_f1": round(indist_f1, 4)}
    mean_indist.append(indist_f1)
    if ts_pos >= 3:
        entry["ts_f1"] = round(ts_f1, 4)
        ood_vals.append(ts_f1)
    if wk_pos >= 3:
        entry["wk_f1"] = round(wk_f1, 4)
        ood_vals.append(wk_f1)
    results.append(entry)
    print(entry, "ts_pos", ts_pos, "wk_pos", wk_pos, flush=True)

    del Xtr, Xtr_s, model; gc.collect()

mean_indist_f1 = sum(mean_indist) / len(mean_indist)
mean_ood_f1 = sum(ood_vals) / len(ood_vals)
print("MEAN_INDIST", round(mean_indist_f1, 4))
print("MEAN_OOD", round(mean_ood_f1, 4))
print("JSON", json.dumps(results))
