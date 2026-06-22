"""Independently reproduce the invention fleet's winner: 'reliable-code surprisal' =
asymmetric cross-entropy trust = log P_reliable(yhat_full). Coverage-match vs the binary
champion and bootstrap the AUGRC margin. Trust nothing until it reproduces here."""
import numpy as np
from idea_harness import H, OOD, _auroc, _augrc
rng = np.random.RandomState(0); B_BOOT = 2000

def surprisal(h):
    rel = h.reliable; relp = h.probe(cols=rel)
    def score(Z):
        pr = np.clip(relp.predict_proba(Z[:, rel])[:, 1], 1e-6, 1-1e-6)
        yhat = h.full.predict(Z)
        return yhat*np.log(pr) + (1-yhat)*np.log(1-pr)   # reliable probe's log-prob of full's label
    return score

print(f"{'pool':6}{'dom':7}{'cov':>6} | {'champ':>7}{'surp@cov':>9}{'Δacc':>7} | "
      f"{'champAUC':>9}{'surpAUC':>8} | {'champGRC':>9}{'surpGRC':>8}{'Δgrc':>8}{'P(s<)':>7}")
rows=[]
for pool in ["mean","max"]:
    h=H(pool); rel=h.reliable; relp=h.probe(cols=rel); sc=surprisal(h)
    for dom,(P,yt) in OOD.items():
        Z=h.std(P[pool].float().numpy()); y=yt.numpy(); fp=h.full.predict(Z); correct=(fp==y).astype(int); n=len(y)
        champ=(fp==relp.predict(Z[:,rel])).astype(float); keep_c=champ>0.5; cov=keep_c.mean(); champ_acc=correct[keep_c].mean()
        s=sc(Z); k=int(round(cov*n)); order=np.argsort(-(s+rng.uniform(0,1e-9,n))); keep_s=np.zeros(n,bool); keep_s[order[:k]]=True
        surp_acc=correct[keep_s].mean()
        cA,sA=_auroc(champ,correct),_auroc(s,correct); cG,sG=_augrc(champ,correct),_augrc(s,correct)
        wins=sum(_augrc(s[bi:=rng.randint(0,n,n)],correct[bi])<_augrc(champ[bi],correct[bi]) for _ in range(B_BOOT)); p=wins/B_BOOT
        rows.append((cA,sA,cG,sG,champ_acc,surp_acc,p))
        print(f"{pool:6}{dom:7}{cov:>6.2f} | {champ_acc:>7.3f}{surp_acc:>9.3f}{surp_acc-champ_acc:>+7.3f} | "
              f"{cA:>9.3f}{sA:>8.3f} | {cG:>9.4f}{sG:>8.4f}{cG-sG:>+8.4f}{p:>7.2f}")
r=np.array(rows)
print(f"\n--- SUMMARY (6 conditions) ---")
print(f"matched-cov selective acc: surprisal beats champion {int((r[:,5]>r[:,4]).sum())}/6 (mean Δ {np.mean(r[:,5]-r[:,4]):+.3f})")
print(f"AUROC: surprisal higher {int((r[:,1]>r[:,0]).sum())}/6 (mean {np.mean(r[:,0]):.3f} -> {np.mean(r[:,1]):.3f})")
print(f"AUGRC: surprisal better {int((r[:,3]<r[:,2]).sum())}/6 (mean margin {np.mean(r[:,2]-r[:,3]):+.4f})")
print(f"paired bootstrap P(surprisal AUGRC<champion): {np.round(r[:,6],2)}; min {r[:,6].min():.2f}")
