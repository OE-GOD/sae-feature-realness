import torch, numpy as np

def f1_score(y, pred, zero_division=0):
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        prec = tp / (tp + fp) if (tp + fp) > 0 else float(zero_division)
        rec = tp / (tp + fn) if (tp + fn) > 0 else float(zero_division)
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

class LogisticRegression:
    def __init__(self, max_iter=2000, C=1.0):
        self.max_iter = max_iter
        self.C = C
    def fit(self, X, y):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        n, d = Xt.shape
        w = torch.zeros(d, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=self.max_iter, line_search_fn="strong_wolfe")
        lam = 1.0 / self.C
        def closure():
            opt.zero_grad()
            z = Xt @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + (lam / (2 * n)) * (w * w).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.w = w.detach(); self.b = b.detach()
        return self
    def decision_function(self, X):
        Xt = torch.tensor(X, dtype=torch.float32)
        return (Xt @ self.w + self.b).numpy()

torch.set_grad_enabled(True)
D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = D["keys"]
NF = 30

def best_thresh(y, scores):
    order = np.argsort(scores)
    ss = scores[order]; ys = y[order]
    # candidate thresholds
    cand = np.unique(ss)
    best_f1, best_t = -1.0, 0.5
    for t in cand:
        pred = (scores >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

train_F = D["train_F"]  # tensor
results = []
indist_list = []
ood_list = []

for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    Xtr = train_F.float().numpy()
    # 1. SELECT corr: point-biserial = pearson corr between feature and binary label
    yc = ytr - ytr.mean()
    Xc = Xtr - Xtr.mean(axis=0)
    num = (Xc * yc[:, None]).sum(axis=0)
    den = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum()) + 1e-12
    corr = num / den
    cols = np.argsort(-np.abs(corr))[:NF]

    Xtr_sel = Xtr[:, cols]
    mu = Xtr_sel.mean(axis=0)
    sd = Xtr_sel.std(axis=0) + 1e-8
    Xtr_std = (Xtr_sel - mu) / sd

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_std, ytr)
    str_tr = clf.decision_function(Xtr_std)
    thr = best_thresh(ytr, str_tr)

    rec = {"concept": C}

    def eval_split(Fkey, Lkey, posmin):
        F = D[Fkey].float().numpy()
        y = D[Lkey][C].numpy().astype(int)
        if y.sum() < posmin:
            return None
        Xs = (F[:, cols] - mu) / sd
        s = clf.decision_function(Xs)
        pred = (s >= thr).astype(int)
        return f1_score(y, pred, zero_division=0)

    indf = eval_split("test_F", "test_L", 1)
    rec["indist_f1"] = float(indf)
    indist_list.append(indf)

    tsf = eval_split("ts_F", "ts_L", 3)
    if tsf is not None:
        rec["ts_f1"] = float(tsf)
    wkf = eval_split("wk_F", "wk_L", 3)
    if wkf is not None:
        rec["wk_f1"] = float(wkf)

    for v in (tsf, wkf):
        if v is not None:
            ood_list.append(v)

    results.append(rec)
    del Xtr, Xtr_sel, Xtr_std
    print(rec)

mean_ind = float(np.mean(indist_list))
mean_ood = float(np.mean(ood_list))
print("MEAN_INDIST", mean_ind)
print("MEAN_OOD", mean_ood)
import json
print("JSON", json.dumps({"mean_indist_f1": mean_ind, "mean_ood_f1": mean_ood, "per_concept": results}))
