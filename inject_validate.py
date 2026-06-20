"""Validate the stability test on REAL Pythia activations with INJECTED ground truth.
Plant known directions into real (superposition-heavy) activations -> oracle on real data."""
import torch, torch.nn as nn
torch.manual_seed(0)
pile=torch.load("/Users/oe/rebuild/thoughts.pt")          # REAL Pythia L6 activations
M,d=pile.shape; scale=pile.std().item()
print(f"real activations {pile.shape}, scale {scale:.3f}")

# plant 15 known unit directions, injected at realistic magnitude
K=15; P=torch.randn(K,d); P=P/P.norm(dim=1,keepdim=True)
inj=pile.clone()
for i in range(M):
    idx=torch.randperm(K)[:3]; coef=(torch.rand(3)*1.0+0.5)*scale*2
    inj[i]=inj[i]+coef@P[idx]

class SAE(nn.Module):
    def __init__(s,d,n=2048,k=32):
        super().__init__(); s.k=k; s.W_enc=nn.Linear(d,n); s.W_dec=nn.Linear(n,d)
    def forward(s,x):
        sc=s.W_enc(x); tk=torch.topk(sc,s.k,-1)
        sp=torch.zeros_like(sc); sp.scatter_(-1,tk.indices,tk.values)
        return s.W_dec(sp),sp
def train(seed):
    torch.manual_seed(seed); m=SAE(d); opt=torch.optim.Adam(m.parameters(),1e-3)
    for ep in range(5):
        perm=torch.randperm(M)
        for i in range(0,M,1024):
            b=inj[perm[i:i+1024]]; r,_=m(b); l=((r-b)**2).mean()
            opt.zero_grad(); l.backward(); opt.step()
    return m
A=train(1); B=train(2)
WdA=A.W_dec.weight.detach(); colsA=WdA/WdA.norm(dim=0)
WdB=B.W_dec.weight.detach(); colsB=WdB/WdB.norm(dim=0)
n=colsA.shape[1]

# RECOVERY: is each planted direction found by SAE-A?
rec=(P@colsA).abs().max(1).values
print(f"\nplanted-direction recovery (max cos to any SAE feature):")
print(f"  {int((rec>0.8).sum())}/{K} planted dirs recovered at cos>0.8 (mean {rec.mean():.3f})")

# GROUND TRUTH per SAE-A feature + STABILITY test
align=(colsA.T@P.T).abs().max(1).values     # how planted-aligned each feature is
is_planted=align>0.6
stab=(colsA.T@colsB).abs().max(1).values
def rate(m,c): m=m.bool(); return c[m].float().mean().item() if m.any() else float('nan')
print(f"\nSTABILITY test on REAL activations w/ injected truth:")
print(f"  planted features found in SAE-A: {int(is_planted.sum())}/{n}")
print(f"  mean stability of PLANTED  : {stab[is_planted].mean():.3f}")
print(f"  mean stability of SPURIOUS : {stab[~is_planted].mean():.3f}")
for TS in (0.7,0.8,0.9):
    print(f"  thresh {TS}: precision {rate(stab>TS,is_planted):.2f}  recall {rate(is_planted,stab>TS):.2f}")
