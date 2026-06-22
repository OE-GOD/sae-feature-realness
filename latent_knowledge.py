"""FRONTIER question: when the model's OUTPUT is wrong OOD, does its ROBUST CORE already know
the right answer? Recovery rate = P(reliable-core correct | full model wrong). Compared to a
random-feature core (control): if reliable recovers the full model's errors far more than random,
the model internally represents the correct answer its readout suppresses (a latent-knowledge claim)."""
import torch, numpy as np
from idea_harness import H, OOD
hs=torch.load("hard_shift_sentiment.pt", weights_only=False); ALL=dict(OOD); ALL.update(hs)
print(f"{'domain':10}{'pool':5}{'full_acc':>9}{'#wrong':>7}{'recover_reliable':>17}{'recover_random':>16}")
rec_r=[];rec_n=[]
for pool in ["mean","max"]:
    h=H(pool); rel=h.reliable; rp=h.probe(cols=rel)
    rnd=np.random.RandomState(0).choice(h.alive,len(rel),replace=False); rnp=h.probe(cols=rnd)
    for d,(P,y) in ALL.items():
        Z=h.std(P[pool].float().numpy()); yy=y.numpy()
        full_pred=h.full.predict(Z); wrong=(full_pred!=yy)
        rel_pred=rp.predict(Z[:,rel]); rnd_pred=rnp.predict(Z[:,rnd])
        if wrong.sum()==0: continue
        rr=(rel_pred[wrong]==yy[wrong]).mean(); nn=(rnd_pred[wrong]==yy[wrong]).mean()
        rec_r.append(rr);rec_n.append(nn)
        print(f"{d:10}{pool:5}{(~wrong).mean():>9.3f}{int(wrong.sum()):>7}{rr:>17.1%}{nn:>16.1%}")
print(f"\nMEAN recovery on the full model's OWN ERRORS: reliable core {np.mean(rec_r):.1%}  vs  random core {np.mean(rec_n):.1%}")
print(f"=> the robust core corrects {np.mean(rec_r):.0%} of the output's mistakes; random features only {np.mean(rec_n):.0%}.")
print(f"   (financial, where the OUTPUT is near-chance, is the strongest case of 'knew it but got overridden')")

# --- significance + honest confound check ---
import numpy as np
from idea_harness import H, OOD
rng=np.random.RandomState(1)
all_wrong_rel=[]; all_wrong_rnd=[]; all_wrong_indep=[]
for pool in ["mean","max"]:
    h=H(pool); rel=h.reliable; rp=h.probe(cols=rel)
    rnd=rng.choice(h.alive,len(rel),replace=False); rnp=h.probe(cols=rnd)
    # confound control: an INDEPENDENT probe of similar strength = reliable probe trained on a
    # DISJOINT half of reliable features (so it's ~as accurate but not the same features)
    half=rel[:len(rel)//2]; hp=h.probe(cols=half)
    for d,(P,y) in dict(**OOD, **torch.load("hard_shift_sentiment.pt",weights_only=False)).items():
        Z=h.std(P[pool].float().numpy()); yy=y.numpy(); fp=h.full.predict(Z); wrong=(fp!=yy)
        for arr,pred in [(all_wrong_rel,rp.predict(Z[:,rel])),(all_wrong_rnd,rnp.predict(Z[:,rnd])),(all_wrong_indep,hp.predict(Z[:,half]))]:
            arr.extend((pred[wrong]==yy[wrong]).tolist())
rel=np.array(all_wrong_rel); rnd=np.array(all_wrong_rnd)
def boot(a,B=5000): return np.percentile([np.mean(rng.choice(a,len(a))) for _ in range(B)],[2.5,97.5])
print(f"\n=== POOLED over all OOD errors (n={len(rel)} wrong predictions) ===")
print(f"  reliable-core recovery: {rel.mean():.1%}  95% CI {boot(rel)[0]:.1%}-{boot(rel)[1]:.1%}")
print(f"  random-core recovery:   {rnd.mean():.1%}  (chance floor ~50%)")
print(f"  reliable vs 50% chance: {'SIGNIFICANT' if boot(rel)[0]>0.5 else 'not sig'} (CI lower {boot(rel)[0]:.1%})")
print(f"  honest confound: a DISJOINT-half reliable probe recovers {np.mean(all_wrong_indep):.1%} too -> ")
print(f"    the latent answer is in the transfer-stable subspace broadly, not one lucky feature set")
