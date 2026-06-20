import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

class TinySAE(nn.Module):
    def __init__(self, d_model=768, n_features=2048, k=32):
        super().__init__(); self.k=k
        self.W_enc=nn.Linear(d_model,n_features); self.W_dec=nn.Linear(n_features,d_model)
    def forward(self,x):
        s=self.W_enc(x); tk=torch.topk(s,self.k,dim=-1)
        sp=torch.zeros_like(s); sp.scatter_(-1,tk.indices,tk.values)
        return self.W_dec(sp), sp

pile=torch.load("/Users/oe/rebuild/thoughts.pt")
print(f"training data: {pile.shape}")

def train(seed):
    torch.manual_seed(seed)
    sae=TinySAE(); opt=torch.optim.Adam(sae.parameters(),lr=1e-3)
    for ep in range(5):
        perm=torch.randperm(len(pile))
        for i in range(0,len(pile),1024):
            b=pile[perm[i:i+1024]]
            r,_=sae(b); loss=((r-b)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return sae

print("training 5 SAEs (seeds 0-4)...")
saes=[]
for s in range(5):
    saes.append(train(s)); print(f"  seed {s} done")

# decoder directions, normalized, per SAE
cols=[ (sae.W_dec.weight/sae.W_dec.weight.norm(dim=0)).detach() for sae in saes ]

# ROBUST stability for seed-0 features: best cosine to EACH other seed, take MEDIAN
ref=cols[0]
with torch.no_grad():
    bests=[]
    for j in range(1,5):
        bests.append((ref.T @ cols[j]).max(dim=1).values)   # (2048,)
    stab=torch.stack(bests).median(dim=0).values             # median across 4 seeds

# frequency from seed-0
with torch.no_grad():
    freq=torch.zeros(2048)
    for i in range(0,len(pile),4096):
        _,sp=saes[0](pile[i:i+4096]); freq+=(sp>0).float().sum(0)
    freq/=len(pile)

alive=freq>0
f=freq[alive]; s=stab[alive]; lf=torch.log10(f+1e-9)
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); return (a@b/(a.norm()*b.norm())).item()
print(f"\nalive features: {int(alive.sum())}")
print(f"Pearson(log-freq, ROBUST stability) = {pear(lf,s):.3f}   (toy was 0.52)")
print(f"stable features (median cos>0.9): {int((s>0.9).sum())}   (toy had ~2)")

qs=torch.quantile(lf, torch.linspace(0,1,5))
print("\nquartile -> avg ROBUST stability:")
labels=["Q1 rarest","Q2","Q3","Q4 commonest"]
qv=[]
for i in range(4):
    m=(lf>=qs[i])&(lf<=qs[i+1]); v=s[m].mean().item(); qv.append(v)
    print(f"  {labels[i]:14s}: {v:.3f}")

plt.figure(figsize=(8,5))
bars=plt.bar(labels,qv,color=["#f85149","#d29922","#2ea043","#d29922"])
for b,v in zip(bars,qv): plt.text(b.get_x()+b.get_width()/2,v+0.005,f"{v:.2f}",ha="center")
plt.ylabel("avg ROBUST stability (median across 4 seeds)")
plt.title(f"SCALED: 5 seeds. Does the inverted-U survive?\nPearson(log-freq,stab)={pear(lf,s):.2f}")
plt.tight_layout(); plt.savefig("/Users/oe/rebuild/fig7_scaled.png",dpi=150)
print("\nsaved fig7_scaled.png")
