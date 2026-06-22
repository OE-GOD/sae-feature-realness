"""CAUSAL test of 'what is it thinking': if spurious topic features CAUSE the OOD errors, then
ABLATING them (zeroing their input to the SAME full probe) should make the readout RECOVER.
Dose-response: ablate the K least-transfer-stable features vs K RANDOM features (control)."""
import torch, numpy as np
from idea_harness import H
hs=torch.load("hard_shift_sentiment.pt", weights_only=False)
print(f"{'pool':5}{'domain':10}{'K_ablated':>10}{'full_acc':>9}{'ablate_unstable':>16}{'ablate_random':>15}")
for pool in ["mean","max"]:
    h=H(pool); w=h.full.coef_[0]; b=h.full.intercept_[0]
    order=np.argsort(h.feat_stability[h.alive])         # least-stable alive features first
    least_stable=h.alive[order]
    rng=np.random.RandomState(0); rand_order=rng.permutation(h.alive)
    for d in ["financial","tweet"]:
        P,y=hs[d]; Z=h.std(P[pool].float().numpy()); yy=y.numpy()
        base=((Z@w+b>0).astype(int)==yy).mean()
        for K in [2000,6000,12000]:
            Zu=Z.copy(); Zu[:,least_stable[:K]]=0; au=((Zu@w+b>0).astype(int)==yy).mean()
            Zr=Z.copy(); Zr[:,rand_order[:K]]=0; ar=((Zr@w+b>0).astype(int)==yy).mean()
            print(f"{pool:5}{d:10}{K:>10}{base:>9.3f}{au:>16.3f}{ar:>15.3f}")
