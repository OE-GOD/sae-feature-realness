import torch, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

NF = 200
d = torch.load('/Users/oe/rebuild/gemma_detector_dataset.pt', map_location='cpu')
concepts = d['keys']

def best_thresh(y, scores):
    order = np.argsort(scores)
    # candidate thresholds
    cands = np.unique(scores)
    if len(cands) > 2000:
        cands = np.quantile(scores, np.linspace(0,1,2000))
    best_f1, best_t = 0.0, 0.5
    for t in cands:
        pred = (scores >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

results = []
indist_list, ood_list = [], []

trainF = d['train_F']  # keep as tensor; index then convert per-concept
for C in concepts:
    ytr = d['train_L'][C].numpy().astype(int)
    if ytr.sum() < 5:
        print(f"skip {C}: <5 pos")
        continue
    Xtr = trainF.float().numpy()  # [9118,16384]

    # 1. L1 selection on all features
    l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)
    l1.fit(Xtr, ytr)
    w = np.abs(l1.coef_.ravel())
    cols = np.argsort(w)[::-1][:NF].copy()
    cols = np.ascontiguousarray(cols)

    Xtr_s = Xtr[:, cols]
    del Xtr
    # 2. standardize via train mean/std
    mu = Xtr_s.mean(0); sd = Xtr_s.std(0); sd[sd==0] = 1.0
    Xtr_z = (Xtr_s - mu)/sd

    # 3. MLP probe
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', alpha=1e-3,
                        max_iter=500, random_state=0)
    clf.fit(Xtr_z, ytr)

    # 4. threshold on train
    str_scores = clf.predict_proba(Xtr_z)[:,1]
    thr = best_thresh(ytr, str_scores)

    rec = {"concept": C}
    # in-dist
    Xte = d['test_F'][:, cols].float().numpy()
    Xte_z = (Xte - mu)/sd
    yte = d['test_L'][C].numpy().astype(int)
    sc = clf.predict_proba(Xte_z)[:,1]
    rec["indist_f1"] = float(f1_score(yte, (sc>=thr).astype(int), zero_division=0))
    indist_list.append(rec["indist_f1"])

    # OOD splits
    ood_vals = []
    for split, Fk, Lk in [("ts","ts_F","ts_L"),("wk","wk_F","wk_L")]:
        yo = d[Lk][C].numpy().astype(int)
        if yo.sum() < 3:
            print(f"skip {C} {split}: <3 pos")
            continue
        Xo = d[Fk][:, cols].float().numpy()
        Xo_z = (Xo - mu)/sd
        sco = clf.predict_proba(Xo_z)[:,1]
        f = float(f1_score(yo, (sco>=thr).astype(int), zero_division=0))
        rec[f"{split}_f1"] = f
        ood_vals.append(f)
    ood_list.extend(ood_vals)
    results.append(rec)
    print(C, rec)

out = {
    "recipe": "l1|nf200|mlp",
    "mean_indist_f1": float(np.mean(indist_list)),
    "mean_ood_f1": float(np.mean(ood_list)),
    "per_concept": results,
}
print(json.dumps(out, indent=2))
with open('/Users/oe/rebuild/result_l1_nf200_mlp.json','w') as f:
    json.dump(out, f, indent=2)
