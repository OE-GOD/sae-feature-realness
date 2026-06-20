"""Supplement: compare BEST non-mean pooling vs mean pooling honestly.
3 seeds, full P/R/F1 on both OOD splits, apples-to-apples pooling swaps."""
import torch, numpy as np, warnings
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
        sel.fit(Xtr_s, ytr); return np.argsort(-np.abs(sel.coef_[0]))[:nf]
    if method == "mi":
        mi = mutual_info_classif(Xtr_s, ytr, random_state=seed); return np.argsort(-mi)[:nf]
    if method == "corr":
        yc = ytr - ytr.mean(); Xc = Xtr_s - Xtr_s.mean(0)
        num = (Xc * yc[:, None]).sum(0); den = np.sqrt((Xc**2).sum(0)*(yc**2).sum())+1e-9
        return np.argsort(-np.abs(num/den))[:nf]
    if method == "meandiff":
        d = Xtr_s[ytr==1].mean(0) - Xtr_s[ytr==0].mean(0); return np.argsort(-np.abs(d))[:nf]

def run_one(pool, method, probe_kind, nf, seed):
    Xtr, ytr = X("train", pool), Y("train")
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1.0
    Z = lambda A:(A-mu)/sd; Xtr_s = Z(Xtr)
    cols = select_cols(Xtr_s, ytr, method, nf, seed)
    probe = LogisticRegression(max_iter=1000, random_state=seed) if probe_kind=="linear" \
            else MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=seed)
    probe.fit(Xtr_s[:, cols], ytr)
    out = {}
    for sp in ["test","rt","am"]:
        Xs = Z(X(sp,pool))[:,cols]; yp = probe.predict(Xs); yt = Y(sp)
        out[sp] = dict(f1=f1_score(yt,yp,zero_division=0), p=precision_score(yt,yp,zero_division=0),
                       r=recall_score(yt,yp,zero_division=0), posrate=float(yp.mean()))
    out["mood"] = (out["rt"]["f1"]+out["am"]["f1"])/2
    return out

def threeseed(pool, m, pk, nf):
    accum = {sp:{k:[] for k in ["f1","p","r","posrate"]} for sp in ["test","rt","am"]}
    mood=[]
    for s in [0,1,2]:
        r = run_one(pool,m,pk,nf,s); mood.append(r["mood"])
        for sp in ["test","rt","am"]:
            for k in ["f1","p","r","posrate"]: accum[sp][k].append(r[sp][k])
    return accum, np.array(mood)

# best combo per pooling family (from phase-1 sweep, seed 0). Re-derive over 3 seeds.
METHODS=["l1","mi","corr","meandiff"]; PROBES=["linear","mlp"]; NFS=[100,400]
print("=== best recipe per pooling family, mean OOD F1 over 3 seeds ===")
best_per_pool={}
for pool in POOLS:
    cands=[]
    for m in METHODS:
        for pk in PROBES:
            for nf in NFS:
                _,mood = threeseed(pool,m,pk,nf)
                cands.append((mood.mean(), mood.std(), (m,pk,nf), mood))
    cands.sort(key=lambda x:-x[0])
    mu,sd,rec,mood = cands[0]; best_per_pool[pool]=(rec,mood)
    print(f"  {pool:9s} best={rec} mOOD mean={mu:.3f} std={sd:.3f} seeds={np.round(mood,3).tolist()}")

print("\n=== FULL P/R/F1 on BOTH OOD splits (3-seed mean+/-std), best recipe per pool ===")
for pool in POOLS:
    rec,_ = best_per_pool[pool]
    acc,mood = threeseed(pool,*rec)
    print(f"\n[{pool}] recipe={rec}")
    for sp in ["rt","am"]:
        a=acc[sp]
        f=np.array(a["f1"]); p=np.array(a["p"]); r=np.array(a["r"]); pr=np.array(a["posrate"])
        print(f"  {sp}: F1={f.mean():.3f}+/-{f.std():.3f}  P={p.mean():.3f}  R={r.mean():.3f}  pred_pos_rate={pr.mean():.3f} (base 0.5)")

print("\n=== APPLES-TO-APPLES: fix mean's best recipe, swap only the pooling ===")
mean_rec,_ = best_per_pool["mean"]
print("recipe (mean's best):", mean_rec)
_,mean_mood = threeseed("mean",*mean_rec)
for pool in POOLS:
    _,mood = threeseed(pool,*mean_rec)
    gap = mood-mean_mood
    print(f"  {pool:9s} mOOD seeds={np.round(mood,3).tolist()} mean={mood.mean():.3f}  gap_vs_mean={gap.mean():+.3f}+/-{gap.std():.3f}")

print("\n=== Does ANY non-mean pooling beat mean's best, under mean's own recipe AND its own best recipe? ===")
mean_best_mood = best_per_pool["mean"][1]
print(f"mean best mOOD = {mean_best_mood.mean():.3f}")
for pool in ["last","max","meanlast3"]:
    rec,mood = best_per_pool[pool]
    gap = mood - mean_best_mood
    verdict = "BEATS mean" if gap.mean()>0.05 else ("within noise" if abs(gap.mean())<=0.05 else "WORSE")
    print(f"  {pool:9s} best mOOD={mood.mean():.3f}  vs mean {mean_best_mood.mean():.3f}  gap={gap.mean():+.3f}+/-{gap.std():.3f}  -> {verdict}")
