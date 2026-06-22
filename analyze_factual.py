"""Does the model internally KNOW the factual truth it judged wrong? Train a mass-mean probe on
transfer-stable features from in-dist topics (cities+companies), test OOD (animals/inventions/
elements). Leak-free: stable feats + mass-mean use only in-dist truth labels."""
import torch, numpy as np
d=torch.load("factual_latent.pt", weights_only=False)
def unit(v): n=np.linalg.norm(v); return v/n if n>1e-9 else v
def scorr(Z,y): yc=y-y.mean(); zc=Z-Z.mean(0); return np.nan_to_num((zc*yc[:,None]).mean(0)/(zc.std(0)*yc.std()+1e-9))
INDIST=["cities","companies"]; OOD=["animals","inventions","elements"]
rng=np.random.RandomState(0)
for pool in ["max","mean"]:
    Xin=np.vstack([d[t][pool].float().numpy() for t in INDIST]); yin=np.concatenate([d[t]["truth"] for t in INDIST])
    mu,sd=Xin.mean(0),Xin.std(0)+1e-6; Zin=(Xin-mu)/sd
    alive=np.where((Xin>0).mean(0)>0.01)[0]
    cc=scorr((d["cities"][pool].float().numpy()-mu)/sd, d["cities"]["truth"])
    co=scorr((d["companies"][pool].float().numpy()-mu)/sd, d["companies"]["truth"])
    stab=cc*co; rel=alive[np.argsort(-stab[alive])[:500]]
    mm=unit(Zin[yin==1][:,rel].mean(0)-Zin[yin==0][:,rel].mean(0)); thr=(Zin[:,rel]@mm).mean()
    rnd=rng.choice(alive,500,replace=False); mmr=unit(Zin[yin==1][:,rnd].mean(0)-Zin[yin==0][:,rnd].mean(0)); thrr=(Zin[:,rnd]@mmr).mean()
    print(f"\n[{pool}] {'topic':11}{'model_acc':>10}{'internal_acc':>13}{'recover_err':>12}{'recover_rand':>13}{'#err':>6}")
    R=[];RM=[]
    for t in OOD:
        Z=(d[t][pool].float().numpy()-mu)/sd; truth=d[t]["truth"]; judge=d[t]["judge"]
        mm_pred=((Z[:,rel]@mm)>thr).astype(int); macc=(judge==truth).mean(); iacc=(mm_pred==truth).mean()
        err=(judge!=truth)
        rec=(mm_pred[err]==truth[err]).mean() if err.sum() else float('nan')
        rndp=((Z[:,rnd]@mmr)>thrr).astype(int); recr=(rndp[err]==truth[err]).mean() if err.sum() else float('nan')
        R.append(rec);RM.append(recr)
        print(f"     {t:11}{macc:>10.3f}{iacc:>13.3f}{rec:>12.1%}{recr:>13.1%}{int(err.sum()):>6}")
    print(f"     {'MEAN':11}{'':>10}{'':>13}{np.nanmean(R):>12.1%}{np.nanmean(RM):>13.1%}")
