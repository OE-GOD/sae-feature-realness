import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)
NF = 400
CONCEPTS = ["newline","comma","period","digit","space_pre","cap_start"]

d = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
train_F = d["train_F"].float()
splits = {
    "indist": (d["test_F"].float(), d["test_L"]),
    "ts":     (d["ts_F"].float(),   d["ts_L"]),
    "wk":     (d["wk_F"].float(),   d["wk_L"]),
}

def best_thresh(y, p):
    order = np.argsort(p)
    cand = np.unique(p)
    # evaluate thresholds at midpoints + extremes
    ths = np.concatenate([[-1e9], (cand[:-1]+cand[1:])/2, [1e9]]) if len(cand)>1 else np.array([0.5])
    bf, bt = -1, 0.5
    for t in ths:
        f = f1_score(y, (p>=t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return bt

per_concept = []
indist_vals, ood_vals = [], []

for C in CONCEPTS:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue

    Xtr = train_F.numpy()

    # 1. L1 selection over all 16384
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000)
    l1.fit(Xtr, ytr)
    w = np.abs(l1.coef_[0])
    cols = np.argsort(w)[::-1][:NF]

    Xtr_s = Xtr[:, cols]
    # 2. standardize with train mean/std
    mu = Xtr_s.mean(0); sd = Xtr_s.std(0); sd[sd==0] = 1.0
    Xtr_z = (Xtr_s - mu) / sd

    # 3. MLP probe
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=300, random_state=0)
    clf.fit(Xtr_z, ytr)

    # 4. threshold tune on train
    ptr = clf.predict_proba(Xtr_z)[:,1]
    t = best_thresh(ytr, ptr)

    rec = {"concept": C}
    for name, (XF, L) in splits.items():
        y = L[C].numpy().astype(int)
        if name != "indist" and y.sum() < 3:
            rec[f"{name}_f1"] = None
            continue
        Xz = (XF.numpy()[:, cols] - mu) / sd
        p = clf.predict_proba(Xz)[:,1]
        f = f1_score(y, (p>=t).astype(int), zero_division=0)
        rec[f"{name}_f1"] = float(f)

    rec_out = {"concept": C, "indist_f1": rec["indist_f1"]}
    if rec.get("ts_f1") is not None: rec_out["ts_f1"] = rec["ts_f1"]
    if rec.get("wk_f1") is not None: rec_out["wk_f1"] = rec["wk_f1"]
    per_concept.append(rec_out)

    indist_vals.append(rec["indist_f1"])
    for nm in ("ts_f1","wk_f1"):
        if rec.get(nm) is not None: ood_vals.append(rec[nm])

    print(C, rec_out, flush=True)

import json
print("MEAN_INDIST", float(np.mean(indist_vals)))
print("MEAN_OOD", float(np.mean(ood_vals)))
print("JSON", json.dumps(per_concept))
