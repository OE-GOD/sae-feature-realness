import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from datasets import load_dataset

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
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k", split="train")

# ---- STABILITY per feature: best cosine of SAE-1 cols vs any SAE-2 col ----
with torch.no_grad():
    W1=sae1.W_dec.weight; W2=sae2.W_dec.weight
    W1n=W1/W1.norm(dim=0); W2n=W2/W2.norm(dim=0)
    stability = (W1n.T @ W2n).max(dim=1).values   # (2048,)

# ---- eval set + which features actually fire ----
docs=[data[i]["text"][:400] for i in range(12)]
fire_count=torch.zeros(2048)
caches=[]
for d in docs:
    toks=model.to_tokens(d)
    with torch.no_grad():
        loss,cache=model.run_with_cache(toks, return_type="loss")
        resid=cache["blocks.6.hook_resid_post"]
        _,sp=sae1(resid[0])
    caches.append((toks, loss.item(), sp))   # sp: (seq,2048)
    fire_count += (sp>0).float().sum(0)

# sample 40 features that DO fire, spanning the stability range
alive=torch.where(fire_count>3)[0]
order=alive[stability[alive].argsort()]
sample=order[torch.linspace(0,len(order)-1,40).long()].tolist()

# ---- CAUSAL per feature: extra loss when ablated across eval set ----
Wd=sae1.W_dec.weight.detach()
causal={}
for f in sample:
    extra=0.0
    col=Wd[:,f]
    for toks,base,sp in caches:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))  # (seq,768)
        def hook(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(toks, return_type="loss",
                                     fwd_hooks=[("blocks.6.hook_resid_post",hook)])
        extra += (abl.item()-base)
    causal[f]=extra/len(caches)

xs=[stability[f].item() for f in sample]
ys=[causal[f] for f in sample]
import statistics
# correlation
mx,my=statistics.mean(xs),statistics.mean(ys)
cov=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
sx=sum((a-mx)**2 for a in xs)**.5; sy=sum((b-my)**2 for b in ys)**.5
r=cov/(sx*sy) if sx*sy>0 else 0

plt.figure(figsize=(9,6))
plt.scatter(xs,ys,c="#1f6feb",s=60,alpha=0.8)
plt.axvline(0.9,color="#f85149",ls="--",label="stable threshold (cos 0.9)")
plt.xlabel("STABILITY  (best cosine to SAE-2)  →  right = stable / replicates")
plt.ylabel("CAUSAL FAITHFULNESS  (extra loss when ablated)  →  up = driver")
plt.title(f"Are stable features the drivers?  (my toy SAE, 40 features)\nPearson r = {r:.2f}")
plt.legend(); plt.tight_layout()
plt.savefig("/Users/oe/rebuild/fig4_axes.png",dpi=150)

print(f"correlation(stability, causal) = {r:.3f}")
print(f"stable features (cos>0.9) sampled: {sum(1 for x in xs if x>0.9)}")
stbl=[y for x,y in zip(xs,ys) if x>0.7]; unst=[y for x,y in zip(xs,ys) if x<0.4]
if stbl: print(f"avg causal of MORE-stable (cos>0.7): {sum(stbl)/len(stbl):.4f}")
if unst: print(f"avg causal of LESS-stable (cos<0.4): {sum(unst)/len(unst):.4f}")
print("saved fig4_axes.png")
