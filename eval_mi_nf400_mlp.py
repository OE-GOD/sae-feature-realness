import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0)
np.random.seed(0)

PATH = "/Users/oe/rebuild/gemma_detector_dataset.pt"
CONCEPTS = ["newline", "comma", "period", "digit", "space_pre", "cap_start"]
NF = 400

data = torch.load(PATH, map_location="cpu")

train_F = data["train_F"].float().numpy()
test_F = data["test_F"].float().numpy()
ts_F = data["ts_F"].float().numpy()
wk_F = data["wk_F"].float().numpy()

def best_threshold(probs, y):
    # tune threshold for best F1 on train
    order = np.unique(probs)
    if len(order) > 512:
        order = np.quantile(probs, np.linspace(0, 1, 512))
    best_t, best_f = 0.5, -1.0
    for t in order:
        f = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

results = {}
for C in CONCEPTS:
    ytr = data["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        print(f"skip {C}: <5 positives in train")
        continue

    # 1. SELECT via mutual info on binarized feat>0
    Xbin = (train_F > 0).astype(int)
    mi = mutual_info_classif(Xbin, ytr, discrete_features=True, random_state=0)
    cols = np.argsort(mi)[::-1][:NF]

    Xtr = train_F[:, cols]
    # 2. standardize using train mean/std
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xtr_s = (Xtr - mu) / sd

    # 3. train MLP probe
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=300, random_state=0)
    clf.fit(Xtr_s, ytr)

    # 4. tune threshold on train
    ptr = clf.predict_proba(Xtr_s)[:, 1]
    thr = best_threshold(ptr, ytr)

    def eval_split(F, Ld, name):
        y = Ld[C].numpy().astype(int)
        if name != "indist" and y.sum() < 3:
            return None
        Xs = (F[:, cols] - mu) / sd
        p = clf.predict_proba(Xs)[:, 1]
        return f1_score(y, (p >= thr).astype(int), zero_division=0)

    indist = eval_split(test_F, data["test_L"], "indist")
    ts = eval_split(ts_F, data["ts_L"], "ts")
    wk = eval_split(wk_F, data["wk_L"], "wk")

    results[C] = {"indist_f1": indist, "ts_f1": ts, "wk_f1": wk}
    print(f"{C}: indist={indist:.4f} ts={ts} wk={wk}")

# means
indist_vals = [r["indist_f1"] for r in results.values()]
ood_vals = []
for r in results.values():
    for k in ("ts_f1", "wk_f1"):
        if r[k] is not None:
            ood_vals.append(r[k])

mean_indist = float(np.mean(indist_vals)) if indist_vals else 0.0
mean_ood = float(np.mean(ood_vals)) if ood_vals else 0.0
print("MEAN_INDIST", mean_indist)
print("MEAN_OOD", mean_ood)

import json
out = {"per_concept": [], "mean_indist_f1": mean_indist, "mean_ood_f1": mean_ood}
for C, r in results.items():
    entry = {"concept": C, "indist_f1": r["indist_f1"]}
    if r["ts_f1"] is not None:
        entry["ts_f1"] = r["ts_f1"]
    if r["wk_f1"] is not None:
        entry["wk_f1"] = r["wk_f1"]
    out["per_concept"].append(entry)
print("JSON", json.dumps(out))
