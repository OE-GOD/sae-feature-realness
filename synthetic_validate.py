"""Validate the certification battery where GROUND TRUTH is known.
Plant true features; check if the battery's tests recover them."""
import torch, torch.nn as nn
torch.manual_seed(0)
d=128; K=20; M=40000
# 20 true feature directions; first 10 are CAUSAL, last 10 INERT
T=torch.randn(K,d); T=T/T.norm(dim=1,keepdim=True)
causal=set(range(10))

# generate activations: each = sparse sum of ~4 true features + noise
acts=torch.zeros(M,d); active=torch.zeros(M,K)
for i in range(M):
    idx=torch.randperm(K)[:4]; coef=torch.rand(4)+0.5
    active[i,idx]=coef
    acts[i]=coef@T[idx]+0.05*torch.randn(d)
# synthetic output driven ONLY by causal features
yproj=T[list(causal)]                      # [10,d]
def y_of(a): return (a@yproj.T).sum(-1)    # output = projection onto causal subspace

class SAE(nn.Module):
    def __init__(s,d,n=64,k=8):
        super().__init__(); s.k=k; s.W_enc=nn.Linear(d,n); s.W_dec=nn.Linear(n,d)
    def forward(s,x):
        sc=s.W_enc(x); tk=torch.topk(sc,s.k,-1)
        sp=torch.zeros_like(sc); sp.scatter_(-1,tk.indices,tk.values)
        return s.W_dec(sp),sp
def train(seed):
    torch.manual_seed(seed); m=SAE(d); opt=torch.optim.Adam(m.parameters(),3e-3)
    for ep in range(8):
        perm=torch.randperm(M)
        for i in range(0,M,512):
            b=acts[perm[i:i+512]]; r,_=m(b); l=((r-b)**2).mean()
            opt.zero_grad(); l.backward(); opt.step()
    return m
A=train(1); B=train(2)
n=A.W_dec.weight.shape[1]
WdA=A.W_dec.weight.detach()                # [d,n]

# GROUND TRUTH per SAE-A feature: best match to a true feature, and its type
colsA=WdA/WdA.norm(dim=0)
sim_true=(colsA.T@T.T)                     # [n,K] cosine to each true feature
best_true=sim_true.abs().max(1).values
best_idx=sim_true.abs().argmax(1)
is_planted=best_true>0.6
is_causal_aligned=torch.tensor([is_planted[f] and best_idx[f].item() in causal for f in range(n)])

# BATTERY test 1 - STABILITY (cross-seed)
WdB=B.W_dec.weight.detach(); colsB=WdB/WdB.norm(dim=0)
stab=(colsA.T@colsB).abs().max(1).values

# BATTERY test 2 - NECESSITY (ablate feature, measure change in causal output y)
with torch.no_grad():
    _,spA=A(acts[:4000])
nec=torch.zeros(n)
base_y=y_of(acts[:4000])
for f in range(n):
    contrib=spA[:,f].unsqueeze(1)*WdA[:,f].unsqueeze(0)
    nec[f]=(y_of(acts[:4000]-contrib)-base_y).abs().mean()

def rate(mask,cond): 
    m=mask.bool(); return (cond[m]).float().mean().item() if m.any() else float('nan')

print("=== VALIDATE certifier on KNOWN ground truth ===")
print(f"planted features found: {int(is_planted.sum())}/{n}   causal-aligned: {int(is_causal_aligned.sum())}\n")
# does STABILITY recover planted features?
print("STABILITY test:")
print(f"  mean stability of PLANTED features : {stab[is_planted].mean():.3f}")
print(f"  mean stability of SPURIOUS features: {stab[~is_planted].mean():.3f}")
TS=0.7
print(f"  precision (stab>{TS} are planted): {rate(stab>TS,is_planted):.2f}")
print(f"  recall    (planted have stab>{TS}): {rate(is_planted,stab>TS):.2f}")
# does NECESSITY recover causal features?
print("\nNECESSITY test (vs known causal features):")
print(f"  mean necessity of CAUSAL-aligned   : {nec[is_causal_aligned].mean():.3f}")
print(f"  mean necessity of inert/spurious   : {nec[~is_causal_aligned].mean():.3f}")
TN=nec[~is_causal_aligned].mean().item()+nec[~is_causal_aligned].std().item()
print(f"  precision (nec>band are causal): {rate(nec>TN,is_causal_aligned):.2f}")
print(f"  recall    (causal have nec>band): {rate(is_causal_aligned,nec>TN):.2f}")
