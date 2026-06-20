import torch, torch.nn as nn

torch.manual_seed(0)
d = torch.load('/Users/oe/rebuild/gemma_detector_dataset.pt', map_location='cpu')
concepts = d['keys']
N_FEATURES = 30

def f1_from_scores(scores, y, thr):
    pred = (scores >= thr).int()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)

def best_threshold(scores, y):
    s = torch.unique(scores)
    # candidate thresholds: midpoints + extremes
    cand = torch.cat([s, torch.tensor([s.min() - 1.0, s.max() + 1.0])])
    best_f1, best_thr = -1.0, 0.0
    for t in cand:
        f = f1_from_scores(scores, y, float(t))
        if f > best_f1:
            best_f1, best_thr = f, float(t)
    return best_thr, best_f1

train_F = d['train_F'].float()
per_concept = []
indist_list, ood_list = [], []

for C in concepts:
    ytr = d['train_L'][C].int()
    if int(ytr.sum()) < 5:
        continue

    # 1. SELECT meandiff: top-k |mean(pos) - mean(neg)|
    pos = train_F[ytr == 1]
    neg = train_F[ytr == 0]
    md = (pos.mean(0) - neg.mean(0)).abs()
    cols = torch.topk(md, N_FEATURES).indices

    Xtr = train_F[:, cols]
    # 2. Standardize using train mean/std
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mu) / sd

    # 3. MLP probe: 1 hidden layer 32 ReLU + light weight decay
    model = nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(), nn.Linear(32, 1))
    yt = ytr.float().unsqueeze(1)
    # class weighting for imbalance
    npos = float((ytr == 1).sum()); nneg = float((ytr == 0).sum())
    pos_weight = torch.tensor([nneg / max(npos, 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    model.train()
    for ep in range(300):
        opt.zero_grad()
        out = model(Xtr_s)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
    model.eval()

    with torch.no_grad():
        train_scores = model(Xtr_s).squeeze(1)
    thr, _ = best_threshold(train_scores, ytr)

    def eval_split(Fkey, Lkey, min_pos=3, require=False):
        y = d[Lkey][C].int()
        if int(y.sum()) < min_pos:
            return None
        X = d[Fkey][:, cols].float()
        Xs = (X - mu) / sd
        with torch.no_grad():
            sc = model(Xs).squeeze(1)
        return f1_from_scores(sc, y, thr)

    indist = eval_split('test_F', 'test_L', min_pos=0)
    ts = eval_split('ts_F', 'ts_L', min_pos=3)
    wk = eval_split('wk_F', 'wk_L', min_pos=3)

    rec = {'concept': C, 'indist_f1': round(indist, 6)}
    if ts is not None:
        rec['ts_f1'] = round(ts, 6); ood_list.append(ts)
    if wk is not None:
        rec['wk_f1'] = round(wk, 6); ood_list.append(wk)
    per_concept.append(rec)
    indist_list.append(indist)
    print(rec)

mean_indist = sum(indist_list) / len(indist_list)
mean_ood = sum(ood_list) / len(ood_list)
print('MEAN_INDIST', round(mean_indist, 6))
print('MEAN_OOD', round(mean_ood, 6))
import json
print('JSON', json.dumps({'mean_indist_f1': mean_indist, 'mean_ood_f1': mean_ood, 'per_concept': per_concept}))
