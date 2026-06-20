import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

d = torch.load('/Users/oe/rebuild/gemma_detector_dataset.pt', map_location='cpu')
keys = d['keys']
N_FEAT = 200

def f1_best_thr(y, scores):
    # tune threshold for best F1
    order = np.argsort(scores)
    thrs = np.unique(scores)
    best = 0.0; bt = 0.0
    # candidate thresholds
    cands = np.concatenate([[scores.min()-1], thrs])
    for t in cands:
        pred = (scores > t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best:
            best = f; bt = t
    return best, bt

train_F = d['train_F']  # keep as is, slice per concept
results = {}
for C in keys:
    ytr = d['train_L'][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    Xtr = train_F.float().numpy()
    # binarize feat>0 for MI
    Xbin = (Xtr > 0).astype(np.int8)
    mi = mutual_info_classif(Xbin, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:N_FEAT]
    Xs = Xtr[:, cols]
    mean = Xs.mean(0); std = Xs.std(0); std[std == 0] = 1.0
    Xstd = (Xs - mean) / std
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xstd, ytr)
    str_tr = clf.decision_function(Xstd)
    _, bt = f1_best_thr(ytr, str_tr)

    def eval_split(Fkey, Lkey):
        XF = d[Fkey].float().numpy()[:, cols]
        y = d[Lkey][C].numpy().astype(int)
        Xs2 = (XF - mean) / std
        sc = clf.decision_function(Xs2)
        pred = (sc > bt).astype(int)
        return y, f1_score(y, pred, zero_division=0)

    yte, f_in = eval_split('test_F', 'test_L')
    yts, f_ts = eval_split('ts_F', 'ts_L')
    ywk, f_wk = eval_split('wk_F', 'wk_L')

    yts_pos = d['ts_L'][C].numpy().sum()
    ywk_pos = d['wk_L'][C].numpy().sum()
    results[C] = dict(indist=f_in,
                      ts=(f_ts if yts_pos >= 3 else None),
                      wk=(f_wk if ywk_pos >= 3 else None))
    del Xtr, Xbin, Xs, Xstd
    print(C, results[C], flush=True)

ind = [r['indist'] for r in results.values()]
ood = []
for r in results.values():
    if r['ts'] is not None: ood.append(r['ts'])
    if r['wk'] is not None: ood.append(r['wk'])
mean_in = float(np.mean(ind))
mean_ood = float(np.mean(ood))
print('MEAN_INDIST', mean_in)
print('MEAN_OOD', mean_ood)
import json
print('JSON', json.dumps({c: r for c, r in results.items()}))
