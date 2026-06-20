import os, torch, torch.nn as nn
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print("loading gemma-2-2b...")
model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
data = load_dataset("NeelNanda/pile-10k", split="train")
HOOK="blocks.12.hook_resid_post"; DM=model.cfg.d_model

# 1. collect real Gemma layer-12 activations
print("collecting Gemma activations...")
vecs=[]
for i in range(120):
    ids=model.to_tokens(data[i]["text"][:500])
    with torch.no_grad():
        _,c=model.run_with_cache(ids, names_filter=[HOOK])
    vecs.append(c[HOOK][0].float().cpu())
    if dev=="mps": torch.mps.empty_cache()
pile=torch.cat(vecs)
print("activations:", tuple(pile.shape))
del model
if dev=="mps": torch.mps.empty_cache()

# 2. train 3 SAEs, same width, different seeds
class SAE(nn.Module):
    def __init__(s,d,n=8192,k=32):
        super().__init__(); s.k=k; s.e=nn.Linear(d,n); s.d=nn.Linear(n,d)
    def forward(s,x):
        sc=s.e(x); tk=torch.topk(sc,s.k,-1)
        sp=torch.zeros_like(sc); sp.scatter_(-1,tk.indices,tk.values)
        return s.d(sp),sp
def train(seed):
    torch.manual_seed(seed); m=SAE(DM); opt=torch.optim.Adam(m.parameters(),1e-3)
    for ep in range(4):
        perm=torch.randperm(len(pile))
        for i in range(0,len(pile),1024):
            b=pile[perm[i:i+1024]]; r,_=m(b); l=((r-b)**2).mean()
            opt.zero_grad(); l.backward(); opt.step()
    return m
saes=[]
for s in range(3):
    saes.append(train(s)); print(f"  SAE seed {s} trained")

# 3. cross-seed stability (median best-cosine across other 2) + 4. frequency
N=8192
cols=[ (m.d.weight/m.d.weight.norm(dim=0)).detach() for m in saes ]
ref=cols[0]
with torch.no_grad():
    b=[ (ref.T@cols[j]).max(1).values for j in (1,2) ]
    stab=torch.stack(b).median(0).values
    freq=torch.zeros(N)
    for i in range(0,len(pile),4096):
        _,sp=saes[0](pile[i:i+4096]); freq+=(sp>0).float().sum(0)
    freq/=len(pile)

alive=freq>0; f=freq[alive]; s=stab[alive]; lf=torch.log10(f+1e-9)
def pear(a,b): a=a-a.mean(); b=b-b.mean(); return (a@b/(a.norm()*b.norm())).item()
print(f"\n=== GEMMA-2-2B, FAIR cross-seed (3 SAEs, width 8k) ===")
print(f"alive: {int(alive.sum())}/{N}")
print(f"Pearson(log-freq, stability) = {pear(lf,s):.3f}   (Pythia toy 0.52, scaled 0.54)")
qs=torch.quantile(lf, torch.linspace(0,1,5))
for i,lab in enumerate(["Q1 rarest","Q2","Q3","Q4 commonest"]):
    m=(lf>=qs[i])&(lf<=qs[i+1]); print(f"  {lab:14s}: {s[m].mean():.3f}")
