"""Adversarial verification of the sentiment pooling+recipe winner.
Sweeps pooling x selection x probe x nf, with 3 seeds, reports P/R/F1 on BOTH OOD splits."""
import torch, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
POOLS = ["mean", "last", "max", "meanlast3"]

def X(split, pool): return data[split][0][pool].float().numpy()
def Y(split): return data[split][1].numpy()

def select_cols(Xtr_s, ytr, method, nf, seed):
    if method == "l1":
        sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000, random_state=seed)
        sel.fit(Xtr_s, ytr)
        return np.argsort(-np.abs(sel.coef_[0]))[:nf]
    if method == "mi":
        mi = mutual_info_classif(Xtr_s, ytr, random_state=seed)
        return np.argsort(-mi)[:nf]
    if method == "corr":
        # abs Pearson corr of each feature with label
        yc = ytr - ytr.mean()
        Xc = Xtr_s - Xtr_s.mean(0)
        num = (Xc * yc[:, None]).sum(0)
        den = np.sqrt((Xc**2).sum(0) * (yc**2).sum()) + 1e-9
        return np.argsort(-np.abs(num/den))[:nf]
    if method == "meandiff":
        d = Xtr_s[ytr==1].mean(0) - Xtr_s[ytr==0].mean(0)
        return np.argsort(-np.abs(d))[:nf]
    raise ValueError(method)

def run_one(pool, method, probe_kind, nf, seed):
    Xtr, ytr = X("train", pool), Y("train")
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0] = 1.0
    Z = lambda A: (A - mu)/sd
    Xtr_s = Z(Xtr)
    cols = select_cols(Xtr_s, ytr, method, nf, seed)
    if probe_kind == "linear":
        probe = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=seed)
    probe.fit(Xtr_s[:, cols], ytr)
    out = {}
    for sp in ["test", "rt", "am"]:
        Xs = Z(X(sp, pool))[:, cols]
        yp = probe.predict(Xs); yt = Y(sp)
        out[sp] = dict(
            f1=f1_score(yt, yp, zero_division=0),
            p=precision_score(yt, yp, zero_division=0),
            r=recall_score(yt, yp, zero_division=0),
            posrate=float(yp.mean()),
        )
    out["mean_ood_f1"] = (out["rt"]["f1"] + out["am"]["f1"]) / 2
    return out

# ---- Phase 1: find winner. Sweep combos, seed 0, mlp+linear, nf in {100,400}
METHODS = ["l1", "mi", "corr", "meandiff"]
PROBES = ["linear", "mlp"]
NFS = [100, 400]
print("=== PHASE 1: SWEEP (seed 0) -> mean OOD F1 ===")
results = {}
rows = []
for pool in POOLS:
    for m in METHODS:
        for pk in PROBES:
            for nf in NFS:
                key = (pool, m, pk, nf)
                r = run_one(pool, m, pk, nf, 0)
                results[key] = r
                rows.append((r["mean_ood_f1"], key, r))
rows.sort(reverse=True)
print(f"{'mOOD_F1':>8} {'inF1':>6} {'rt_F1':>6} {'am_F1':>6}  combo")
for mood, key, r in rows[:12]:
    print(f"{mood:8.3f} {r['test']['f1']:6.3f} {r['rt']['f1']:6.3f} {r['am']['f1']:6.3f}  {key}")

winner = rows[0][1]
print("\nWINNER (by mean OOD F1, seed 0):", winner)

# mean-pooling baseline: best mean-pool combo
mean_rows = sorted([(r["mean_ood_f1"], k, r) for (mood,k,r) in [(x[0],x[1],x[2]) for x in rows] if k[0]=="mean"], reverse=True)
best_mean = mean_rows[0][1]
print("BEST mean-pooling combo:", best_mean, "mOOD_F1=%.3f" % mean_rows[0][0])

print("\n=== PHASE 2: 3-SEED STABILITY ===")
def threeseed(key):
    pool, m, pk, nf = key
    vals = {"test":[], "rt":[], "am":[], "mood":[]}
    detail = []
    for s in [0,1,2]:
        r = run_one(pool, m, pk, nf, s)
        vals["test"].append(r["test"]["f1"]); vals["rt"].append(r["rt"]["f1"])
        vals["am"].append(r["am"]["f1"]); vals["mood"].append(r["mean_ood_f1"])
        detail.append(r)
    return vals, detail

for label, key in [("WINNER", winner), ("BEST-MEAN", best_mean)]:
    vals, detail = threeseed(key)
    print(f"\n[{label}] {key}")
    for sp in ["test","rt","am","mood"]:
        a = np.array(vals[sp]);
        nm = "mean_OOD_F1" if sp=="mood" else f"{sp}_F1"
        print(f"  {nm:>12}: seeds={np.round(a,3).tolist()}  mean={a.mean():.3f} std={a.std():.3f} range={a.max()-a.min():.3f}")
    # P/R/posrate on OOD splits, seed 0
    r0 = detail[0]
    for sp in ["rt","am"]:
        d = r0[sp]
        print(f"  {sp}: P={d['p']:.3f} R={d['r']:.3f} F1={d['f1']:.3f} pred_pos_rate={d['posrate']:.3f} (base rate 0.5)")

print("\n=== PHASE 3: winner vs mean-pool, per-seed gap ===")
vw,_ = threeseed(winner); vm,_ = threeseed(best_mean)
gaps = np.array(vw["mood"]) - np.array(vm["mood"])
print("winner mOOD per seed:", np.round(vw["mood"],3).tolist())
print("mean   mOOD per seed:", np.round(vm["mood"],3).tolist())
print("gap per seed:", np.round(gaps,3).tolist(), "mean gap=%.3f std=%.3f"%(gaps.mean(),gaps.std()))

# Also compare winner against the SAME recipe but mean pooling (apples-to-apples)
print("\n=== PHASE 3b: winner recipe with mean pooling (same selection/probe/nf) ===")
_, m, pk, nf = winner
vw2,_ = threeseed(("mean", m, pk, nf))
gaps2 = np.array(vw["mood"]) - np.array(vw2["mood"])
print(f"recipe={m}/{pk}/nf={nf}: winner-pool({winner[0]}) mOOD={np.round(vw['mood'],3).tolist()}")
print(f"                         mean-pool mOOD={np.round(vw2['mood'],3).tolist()}")
print("apples-to-apples gap per seed:", np.round(gaps2,3).tolist(), "mean=%.3f std=%.3f"%(gaps2.mean(),gaps2.std()))
