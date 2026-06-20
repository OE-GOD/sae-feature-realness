import torch

torch.manual_seed(0)
DATA = "/Users/oe/rebuild/gemma_detector_dataset.pt"
CONCEPTS = ["newline", "comma", "period", "digit", "space_pre", "cap_start"]
NF = 400

d = torch.load(DATA, map_location="cpu")

train_F = d["train_F"].float()
test_F = d["test_F"].float()
ts_F = d["ts_F"].float()
wk_F = d["wk_F"].float()


def f1_at_threshold(scores, y, thr):
    pred = (scores >= thr).float()
    tp = (pred * y).sum()
    fp = (pred * (1 - y)).sum()
    fn = ((1 - pred) * y).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return torch.tensor(0.0)
    return 2 * tp / denom


def best_threshold(scores, y):
    # candidate thresholds from sorted unique scores
    s = torch.unique(scores)
    if s.numel() > 2000:
        idx = torch.linspace(0, s.numel() - 1, 2000).long()
        s = s[idx]
    cands = torch.cat([s - 1e-6, (s[:-1] + s[1:]) / 2]) if s.numel() > 1 else s
    best_f1, best_thr = -1.0, 0.0
    for thr in cands:
        f = f1_at_threshold(scores, y, thr).item()
        if f > best_f1:
            best_f1, best_thr = f, thr.item()
    return best_thr, best_f1


def train_logreg(X, y, epochs=300, lr=0.5, wd=1e-4):
    n, dfeat = X.shape
    w = torch.zeros(dfeat, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = y.sum()
    neg = n - pos
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=100.0)
    opt = torch.optim.Adam([w, b], lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(epochs):
        opt.zero_grad()
        logits = X @ w + b
        loss = lossf(logits, y) + wd * (w * w).sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


results = {}
for C in CONCEPTS:
    ytr = d["train_L"][C].float()
    if ytr.sum() < 5:
        continue

    # 1. SELECT top-k by |point-biserial corr| on train_F
    Xtr_all = train_F
    mu = Xtr_all.mean(0)
    sd = Xtr_all.std(0).clamp(min=1e-8)
    Xz_all = (Xtr_all - mu) / sd
    y_c = (ytr - ytr.mean()) / ytr.std().clamp(min=1e-8)
    corr = (Xz_all * y_c.unsqueeze(1)).mean(0).abs()  # point-biserial = corr of standardized
    sel = torch.topk(corr, NF).indices

    # 2. Standardize selected using train mean/std
    smu = mu[sel]
    ssd = sd[sel]
    Xtr = (train_F[:, sel] - smu) / ssd

    # 3. Train linear probe
    w, b = train_logreg(Xtr, ytr)

    # 4. Tune threshold on TRAIN
    tr_scores = Xtr @ w + b
    thr, tr_f1 = best_threshold(tr_scores, ytr)

    # 5. Evaluate
    def eval_split(F, L, min_pos=1):
        y = L.float()
        if y.sum() < min_pos:
            return None
        Xs = (F[:, sel] - smu) / ssd
        sc = Xs @ w + b
        return f1_at_threshold(sc, y, thr).item()

    indist = eval_split(test_F, d["test_L"][C], min_pos=1)
    ts = eval_split(ts_F, d["ts_L"][C], min_pos=3)
    wk = eval_split(wk_F, d["wk_L"][C], min_pos=3)

    results[C] = {"indist_f1": indist, "ts_f1": ts, "wk_f1": wk, "train_f1": tr_f1}
    print(C, "indist=%.4f" % indist,
          "ts=%s" % (("%.4f" % ts) if ts is not None else "skip"),
          "wk=%s" % (("%.4f" % wk) if wk is not None else "skip"))

# Means
indist_vals = [r["indist_f1"] for r in results.values() if r["indist_f1"] is not None]
ood_vals = []
for r in results.values():
    for k in ("ts_f1", "wk_f1"):
        if r[k] is not None:
            ood_vals.append(r[k])

mean_indist = sum(indist_vals) / len(indist_vals)
mean_ood = sum(ood_vals) / len(ood_vals)
print("MEAN_INDIST %.4f" % mean_indist)
print("MEAN_OOD %.4f" % mean_ood)
import json
print("JSON", json.dumps(results))
