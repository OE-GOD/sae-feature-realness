import torch, torch.nn as nn
from collections import Counter
from transformer_lens import HookedTransformer
from datasets import load_dataset

class TinySAE(nn.Module):
    def __init__(s,d=768,n=2048,k=32):
        super().__init__(); s.k=k; s.W_enc=nn.Linear(d,n); s.W_dec=nn.Linear(n,d)
    def forward(s,x):
        sc=s.W_enc(x); tk=torch.topk(sc,s.k,-1)
        sp=torch.zeros_like(sc); sp.scatter_(-1,tk.indices,tk.values)
        return s.W_dec(sp),sp

sae=TinySAE(); sae.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae.eval()
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k",split="train")
N=2048; Wd=sae.W_dec.weight.detach()

# collect docs: baseline loss, sparse acts, token ids
docs=[]; firetok=[Counter() for _ in range(N)]
for i in range(15):
    ids=model.to_tokens(data[i]["text"][:600])
    with torch.no_grad():
        base,c=model.run_with_cache(ids,return_type="loss")
        _,sp=sae(c["blocks.6.hook_resid_post"][0])
    docs.append((ids,base.item(),sp))
    tid=ids[0].tolist(); rows,cols=torch.where(sp>0)
    for r,cc in zip(rows.tolist(),cols.tolist()): firetok[cc][tid[r]]+=1

interp=torch.zeros(N); fires=torch.zeros(N)
for f in range(N):
    c=firetok[f]; tot=sum(c.values()); fires[f]=tot
    if tot>0: interp[f]=sum(v for _,v in c.most_common(3))/tot

# sample 50 features spanning interpretability, that fire enough
cand=torch.where(fires>=8)[0]
order=cand[interp[cand].argsort()]
sample=order[torch.linspace(0,len(order)-1,50).long()].tolist()

xs=[]; ys=[]  # interpretability , per-firing causal
for f in sample:
    col=Wd[:,f]; tot=0.0; nf=0
    for ids,base,sp in docs:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[("blocks.6.hook_resid_post",hk)])
        tot+=(abl.item()-base); nf+=int((sp[:,f]>0).sum())
    if nf>0: xs.append(interp[f].item()); ys.append(tot/nf)

import statistics
def pear(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    sa=sum((x-ma)**2 for x in a)**.5; sb=sum((y-mb)**2 for y in b)**.5
    return cov/(sa*sb) if sa*sb else 0
print(f"features: {len(xs)}")
print(f"Pearson(interpretability, per-firing CAUSAL) = {pear(xs,ys):.3f}")
print("\n--- the triangle of 'real' ---")
print(f"  stability  vs causation       ~ -0.13  (independent)")
print(f"  stability  vs interpretability~ +0.00  (independent)")
print(f"  interpretability vs causation = {pear(xs,ys):.3f}")
