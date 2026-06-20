import torch
import numpy as np

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float().numpy()
test_F = d["test_F"].float().numpy()
ood_F = d["ood_F"].float().numpy()
train_L = d["train_L"]
test_L = d["test_L"]
ood_L = d["ood_L"]

n_features = 30

def f1_score(y, pred):
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def best_threshold(scores, y):
    order = np.argsort(scores)
    cands = np.unique(scores)
    # try midpoints + extremes
    thrs = np.concatenate([[scores.min() - 1e-6], (cands[:-1] + cands[1:]) / 2.0, [scores.max() + 1e-6]]) if len(cands) > 1 else np.array([scores.min() - 1e-6, scores.max() + 1e-6])
    best_t, best_f = thrs[0], -1.0
    for t in thrs:
        f = f1_score(y, (scores >= t).astype(int))
        if f > best_f:
            best_f, best_t = f, t
    return best_t

per_concept = []
ind_f1s, ood_f1s = [], []

for C in keys:
    ytr = train_L[C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    yte = test_L[C].numpy().astype(int)
    yood = ood_L[C].numpy().astype(int)

    # meandiff selection
    pos = train_F[ytr == 1]
    neg = train_F[ytr == 0]
    diff = np.abs(pos.mean(axis=0) - neg.mean(axis=0))
    cols = np.argsort(diff)[::-1][:n_features]

    Xtr = train_F[:, cols]
    Xte = test_F[:, cols]
    Xood = ood_F[:, cols]

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xood = (Xood - mu) / sd

    # logistic regression via gradient descent (BCE), torch
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
    lossfn = torch.nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        logit = Xt @ w + b
        loss = lossfn(logit, yt) + 1e-4 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    w_ = w.detach().numpy(); b_ = float(b.detach().numpy()[0])

    str_tr = Xtr @ w_ + b_
    t = best_threshold(str_tr, ytr)

    str_te = Xte @ w_ + b_
    str_ood = Xood @ w_ + b_

    f_ind = f1_score(yte, (str_te >= t).astype(int))
    f_ood = f1_score(yood, (str_ood >= t).astype(int))

    per_concept.append((C, f_ind, f_ood))
    ind_f1s.append(f_ind)
    ood_f1s.append(f_ood)

print("mean_indist_f1", np.mean(ind_f1s))
print("mean_ood_f1", np.mean(ood_f1s))
for c, fi, fo in per_concept:
    print(c, fi, fo)
