import torch
import torch.nn as nn
import json

torch.manual_seed(0)
DEV = "cpu"
N_FEATURES = 30

d = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = d["keys"]

train_F = d["train_F"]
test_F = d["test_F"]
ts_F = d["ts_F"]
wk_F = d["wk_F"]


def f1_score(y_true, y_pred):
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def best_threshold_f1(scores, y):
    uniq = torch.unique(scores)
    if uniq.numel() > 1:
        cands = (uniq[:-1] + uniq[1:]) / 2
        cands = torch.cat([uniq[:1] - 1e-3, cands, uniq[-1:] + 1e-3])
    else:
        cands = torch.tensor([uniq[0] - 1e-3, uniq[0] + 1e-3])
    best_f1, best_t = -1.0, 0.0
    for t in cands:
        pred = (scores >= t).int()
        f1 = f1_score(y, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def corr_select(X, y, k):
    Xf = X.float()
    yf = y.float()
    xm = Xf.mean(0)
    xs = Xf.std(0) + 1e-8
    ym = yf.mean()
    ys = yf.std() + 1e-8
    cov = ((Xf - xm) * (yf - ym).unsqueeze(1)).mean(0)
    corr = cov / (xs * ys)
    corr = torch.nan_to_num(corr, nan=0.0)
    idx = torch.argsort(corr.abs(), descending=True)[:k]
    return idx


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y):
    model = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    pos = float(y.sum())
    neg = float((y == 0).sum())
    pos_weight = torch.tensor(max(neg / max(pos, 1), 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    yf = y.float()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(model(X), yf)
        loss.backward()
        opt.step()
    return model


results = {}
indist_list = []
ood_list = []

for C in concepts:
    ytr = d["train_L"][C].int()
    if int(ytr.sum()) < 5:
        continue

    idx = corr_select(train_F, ytr, N_FEATURES)

    Xtr = train_F[:, idx].float()
    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mean) / std

    model = train_mlp(Xtr_s, ytr)
    model.eval()
    with torch.no_grad():
        tr_logits = model(Xtr_s)
    thr, _ = best_threshold_f1(tr_logits, ytr)

    def eval_split(F, L):
        y = L[C].int()
        if int(y.sum()) < 3:
            return None
        Xs = (F[:, idx].float() - mean) / std
        with torch.no_grad():
            logits = model(Xs)
        pred = (logits >= thr).int()
        return f1_score(y, pred)

    indist = eval_split(test_F, d["test_L"])
    ts = eval_split(ts_F, d["ts_L"])
    wk = eval_split(wk_F, d["wk_L"])

    results[C] = {"indist_f1": indist, "ts_f1": ts, "wk_f1": wk}
    indist_list.append(indist)
    for v in (ts, wk):
        if v is not None:
            ood_list.append(v)

    print(C, "indist=%.4f" % indist,
          "ts=%s" % ("%.4f" % ts if ts is not None else "SKIP"),
          "wk=%s" % ("%.4f" % wk if wk is not None else "SKIP"))

mean_indist = sum(indist_list) / len(indist_list)
mean_ood = sum(ood_list) / len(ood_list)
print("MEAN_INDIST=%.4f" % mean_indist)
print("MEAN_OOD=%.4f" % mean_ood)
print("JSON_RESULTS=" + json.dumps(results))
