import torch, numpy as np, json

def f1_score(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)

class LogReg:
    def __init__(self, lr=0.5, n_iter=2000, l2=1e-4):
        self.lr = lr; self.n_iter = n_iter; self.l2 = l2
    def fit(self, X, y):
        n, dd = X.shape
        self.w = np.zeros(dd); self.b = 0.0
        y = y.astype(np.float64)
        pos = y.sum(); neg = n - pos
        wpos = n / (2 * pos) if pos > 0 else 1.0
        wneg = n / (2 * neg) if neg > 0 else 1.0
        sw = np.where(y == 1, wpos, wneg)
        for _ in range(self.n_iter):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y) * sw
            gw = X.T @ g / n + self.l2 * self.w
            gb = g.mean()
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self
    def decision_function(self, X):
        return X @ self.w + self.b

def best_thr_f1(y, scores):
    thrs = np.unique(scores)
    if thrs.size > 2000:
        thrs = np.quantile(scores, np.linspace(0, 1, 2000))
    best_f, best_t = -1, 0.0
    for t in thrs:
        f = f1_score(y, (scores >= t).astype(int))
        if f > best_f:
            best_f, best_t = f, t
    return best_t

d = torch.load('/Users/oe/rebuild/gemma_detector_dataset.pt', map_location='cpu')
concepts = d['keys'] if isinstance(d.get('keys'), list) else list(d['train_L'].keys())

train_F = d['train_F'].float().numpy()
test_F = d['test_F'].float().numpy()
ts_F = d['ts_F'].float().numpy()
wk_F = d['wk_F'].float().numpy()

NF = 100

results = {}
indist_list = []
ood_list = []

for C in concepts:
    ytr = d['train_L'][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    # SELECTION: corr = top-k |point-biserial| (pearson corr feat vs binary label)
    y = ytr.astype(np.float64)
    yc = (y - y.mean()) / (y.std() + 1e-12)
    fmean = train_F.mean(0)
    fstd = train_F.std(0); fstd[fstd == 0] = 1.0
    Xc = (train_F - fmean) / fstd
    corr = np.abs((Xc * yc[:, None]).mean(0))
    cols = np.argsort(corr)[::-1][:NF]

    Xtr = train_F[:, cols]
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd

    clf = LogReg().fit(Xtr_s, ytr)
    thr = best_thr_f1(ytr, clf.decision_function(Xtr_s))

    def eval_split(F, L):
        yy = L.numpy().astype(int)
        if yy.sum() < 3:
            return None
        Xs = (F[:, cols] - mu) / sd
        sc = clf.decision_function(Xs)
        return f1_score(yy, (sc >= thr).astype(int))

    indist = eval_split(test_F, d['test_L'][C])
    ts = eval_split(ts_F, d['ts_L'][C])
    wk = eval_split(wk_F, d['wk_L'][C])

    results[C] = dict(indist_f1=indist, ts_f1=ts, wk_f1=wk)
    if indist is not None:
        indist_list.append(indist)
    for v in (ts, wk):
        if v is not None:
            ood_list.append(v)

mean_indist = float(np.mean(indist_list))
mean_ood = float(np.mean(ood_list))

out = {"recipe": "corr|nf100|linear",
       "mean_indist_f1": mean_indist,
       "mean_ood_f1": mean_ood,
       "results": results}
print("JSON_START")
print(json.dumps(out))
print("JSON_END")
