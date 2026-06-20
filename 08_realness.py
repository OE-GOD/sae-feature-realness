import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

class TinySAE(nn.Module):
    def __init__(self, d_model=768, n_features=2048, k=32):
        super().__init__(); self.k=k
        self.W_enc=nn.Linear(d_model,n_features); self.W_dec=nn.Linear(n_features,d_model)
    def forward(self,x):
        s=self.W_enc(x); tk=torch.topk(s,self.k,dim=-1)
        sp=torch.zeros_like(s); sp.scatter_(-1,tk.indices,tk.values)
        return self.W_dec(sp), sp

sae1=TinySAE(); sae1.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae1.eval()
sae2=TinySAE(); sae2.load_state_dict(torch.load("/Users/oe/rebuild/sae2.pt")); sae2.eval()
pile=torch.load("/Users/oe/rebuild/thoughts.pt")   # 69529 x 768 thoughts

# CHEAP signal: firing frequency per feature (one SAE only)
with torch.no_grad():
    freq=torch.zeros(2048)
    for i in range(0,len(pile),4096):
        _,sp=sae1(pile[i:i+4096])
        freq+=(sp>0).float().sum(0)
    freq=freq/len(pile)   # fraction of tokens each feature fires on

# EXPENSIVE ground truth: stability (needs a 2nd SAE)
    W1=sae1.W_dec.weight; W2=sae2.W_dec.weight
    W1n=W1/W1.norm(dim=0); W2n=W2/W2.norm(dim=0)
    stability=(W1n.T @ W2n).max(dim=1).values

alive=freq>0
f=freq[alive]; s=stability[alive]
lf=torch.log10(f+1e-9)   # frequency spans orders of magnitude -> log

def pearson(a,b):
    a=a-a.mean(); b=b-b.mean()
    return (a@b/(a.norm()*b.norm())).item()

r_raw=pearson(f,s); r_log=pearson(lf,s)
print(f"alive features: {int(alive.sum())}/2048")
print(f"Pearson(frequency, stability)      = {r_raw:.3f}")
print(f"Pearson(log-frequency, stability)  = {r_log:.3f}")

# binned view: avg stability for rare vs common features
import numpy as np
qs=torch.quantile(lf, torch.tensor([0.0,0.25,0.5,0.75,1.0]))
print("\nfrequency quartile  ->  avg stability:")
for i in range(4):
    m=(lf>=qs[i])&(lf<=qs[i+1])
    print(f"  Q{i+1} ({'rarest' if i==0 else 'commonest' if i==3 else '...'}):  avg stability {s[m].mean():.3f}")

plt.figure(figsize=(9,6))
plt.scatter(lf.numpy(), s.numpy(), s=8, alpha=0.3, c="#1f6feb")
plt.axhline(0.9, color="#f85149", ls="--", label="stable threshold")
plt.xlabel("log10 firing frequency  (CHEAP, one SAE)  ->  right = fires often")
plt.ylabel("stability  (EXPENSIVE ground truth)  ->  up = replicates")
plt.title(f"Can cheap firing-frequency predict expensive stability?\nPearson(log-freq, stability) = {r_log:.2f}")
plt.legend(); plt.tight_layout()
plt.savefig("/Users/oe/rebuild/fig5_realness.png", dpi=150)
print("\nsaved fig5_realness.png")
