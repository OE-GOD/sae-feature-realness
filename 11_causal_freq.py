import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from datasets import load_dataset

class TinySAE(nn.Module):
    def __init__(self,d=768,n=2048,k=32):
        super().__init__(); self.k=k
        self.W_enc=nn.Linear(d,n); self.W_dec=nn.Linear(n,d)
    def forward(self,x):
        s=self.W_enc(x); tk=torch.topk(s,self.k,dim=-1)
        sp=torch.zeros_like(s); sp.scatter_(-1,tk.indices,tk.values)
        return self.W_dec(sp),sp

sae=TinySAE(); sae.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae.eval()
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k",split="train")
pile=torch.load("/Users/oe/rebuild/thoughts.pt")

# frequency (cheap)
with torch.no_grad():
    freq=torch.zeros(2048)
    for i in range(0,len(pile),4096):
        _,sp=sae(pile[i:i+4096]); freq+=(sp>0).float().sum(0)
    freq/=len(pile)

# eval corpus: cache resid + sparse per doc, and baseline loss
docs=[]
for i in range(15):
    ids=model.to_tokens(data[i]["text"][:600])
    with torch.no_grad():
        base,c=model.run_with_cache(ids,return_type="loss")
        _,sp=sae(c["blocks.6.hook_resid_post"][0])
    docs.append((ids,base.item(),sp))

# sample 50 features spanning frequency, that fire in eval
firecorpus=torch.zeros(2048)
for _,_,sp in docs: firecorpus+=(sp>0).float().sum(0)
cand=torch.where(firecorpus>=5)[0]
order=cand[freq[cand].argsort()]
sample=order[torch.linspace(0,len(order)-1,50).long()].tolist()

Wd=sae.W_dec.weight.detach()
xs=[]; ys=[]  # log-freq , per-firing causal impact
for fidx in sample:
    col=Wd[:,fidx]; tot=0.0; fires=0
    for ids,base,sp in docs:
        contrib=(sp[:,fidx].unsqueeze(1)*col.unsqueeze(0))
        def abl_hook(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",
                  fwd_hooks=[("blocks.6.hook_resid_post",abl_hook)])
        tot += (abl.item()-base); fires += int((sp[:,fidx]>0).sum())
    if fires>0:
        xs.append(torch.log10(freq[fidx]+1e-9).item())
        ys.append(tot/fires)   # PER-FIRING impact (confound controlled)

import statistics
def pear(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    sa=sum((x-ma)**2 for x in a)**.5; sb=sum((y-mb)**2 for y in b)**.5
    return cov/(sa*sb) if sa*sb else 0
r=pear(xs,ys)
print(f"features tested: {len(xs)}")
print(f"Pearson(log-freq, PER-FIRING causal impact) = {r:.3f}")
# also show the confound we AVOIDED: raw total would be...
print("(reminder: raw-total causal would trivially track frequency; we used per-firing)")

plt.figure(figsize=(8,5))
plt.scatter(xs,ys,s=40,c="#1f6feb",alpha=0.8)
plt.xlabel("log10 firing frequency"); plt.ylabel("per-firing causal impact (loss/firing)")
plt.title(f"Does frequency predict per-firing causal importance?\nPearson = {r:.2f} (confound controlled)")
plt.tight_layout(); plt.savefig("/Users/oe/rebuild/fig8_causal_freq.png",dpi=150)
print("saved fig8_causal_freq.png")
