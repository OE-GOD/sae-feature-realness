import torch, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

NF = 100
d = torch.load('/Users/oe/rebuild/gemma_detector_dataset.pt', map_location='cpu')
concepts = d['keys'] if isinstance(d['keys'], list) else list(d['train_L'].keys())

trF = d['train_F'].float().numpy()
teF = d['test_F'].float().numpy()
tsF = d['ts_F'].float().numpy()
wkF = d['wk_F'].float().numpy()

def best_thr(y, scores):
    cand = np.unique(scores)
    ths = np.concatenate([[cand.min()-1], (cand[:-1]+cand[1:])/2, [cand.max()+1]])
    bf, bt = -1, 0.0
    for t in ths:
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return bt

results = []
for C in concepts:
    ytr = d['train_L'][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    pos = trF[ytr == 1]; neg = trF[ytr == 0]
    md = np.abs(pos.mean(0) - neg.mean(0))
    cols = np.argsort(md)[::-1][:NF]
    Xtr = trF[:, cols]
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr_s, ytr)
    thr = best_thr(ytr, clf.decision_function(Xtr_s))

    def ev(F, L):
        y = L.numpy().astype(int)
        Xs = (F[:, cols] - mu) / sd
        sc = clf.decision_function(Xs)
        return float(f1_score(y, (sc >= thr).astype(int), zero_division=0)), int(y.sum())

    ind, _ = ev(teF, d['test_L'][C])
    ts, tsp = ev(tsF, d['ts_L'][C])
    wk, wkp = ev(wkF, d['wk_L'][C])
    row = {'concept': C, 'indist_f1': round(ind, 4)}
    if tsp >= 3: row['ts_f1'] = round(ts, 4)
    if wkp >= 3: row['wk_f1'] = round(wk, 4)
    results.append(row)

mi = float(np.mean([r['indist_f1'] for r in results]))
ood = [r[k] for r in results for k in ('ts_f1','wk_f1') if k in r]
mo = float(np.mean(ood))
print(json.dumps({'recipe':'meandiff|nf100|linear','mean_indist_f1':round(mi,4),
                  'mean_ood_f1':round(mo,4),'per_concept':results}, indent=2))
