import torch, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

torch.manual_seed(0); np.random.seed(0)
D = torch.load("/Users/oe/rebuild/gemma_detector_dataset.pt", map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]
NF = 100

def best_thr(y, p):
    bt, bf = 0.5, -1.0
    for t in np.unique(p):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return bt

trainF = D["train_F"]  # tensor
results = []
indist_list, ood_list = [], []

for C in concepts:
    ytr = D["train_L"][C].numpy().astype(int)
    if ytr.sum() < 5:
        continue
    Xtr = trainF.float().numpy()
    # 1. corr selection: point-biserial == pearson corr between binary y and feature
    yc = ytr - ytr.mean()
    Xc = Xtr - Xtr.mean(0)
    num = (Xc * yc[:, None]).sum(0)
    den = np.sqrt((Xc**2).sum(0) * (yc**2).sum()) + 1e-12
    corr = num / den
    cols = np.argsort(-np.abs(corr))[:NF]

    Xtr_s = Xtr[:, cols]
    scaler = StandardScaler().fit(Xtr_s)
    Xtr_z = scaler.transform(Xtr_s)

    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                        alpha=1e-3, max_iter=300, random_state=0)
    clf.fit(Xtr_z, ytr)
    ptr = clf.predict_proba(Xtr_z)[:, 1]
    thr = best_thr(ytr, ptr)

    def evalsplit(Fkey, Lkey):
        y = D[Lkey][C].numpy().astype(int)
        X = D[Fkey].float().numpy()[:, cols]
        Xz = scaler.transform(X)
        p = clf.predict_proba(Xz)[:, 1]
        return y, f1_score(y, (p >= thr).astype(int), zero_division=0)

    yte, f_in = evalsplit("test_F", "test_L")

    yts = D["ts_L"][C].numpy().astype(int)
    ywk = D["wk_L"][C].numpy().astype(int)
    ts_f = None; wk_f = None
    if yts.sum() >= 3:
        _, ts_f = evalsplit("ts_F", "ts_L")
    if ywk.sum() >= 3:
        _, wk_f = evalsplit("wk_F", "wk_L")

    rec = {"concept": C, "indist_f1": float(f_in)}
    indist_list.append(f_in)
    if ts_f is not None:
        rec["ts_f1"] = float(ts_f); ood_list.append(ts_f)
    if wk_f is not None:
        rec["wk_f1"] = float(wk_f); ood_list.append(wk_f)
    results.append(rec)
    print(rec)
    del Xtr, Xtr_s, Xtr_z

print("MEAN_INDIST", float(np.mean(indist_list)))
print("MEAN_OOD", float(np.mean(ood_list)))
import json
print("JSON", json.dumps(results))
