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
sae2=TinySAE(); sae2.load_state_dict(torch.load("/Users/oe/rebuild/sae2.pt")); sae2.eval()
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k",split="train")
N=2048; Wd=sae.W_dec.weight.detach()

# LOGIT COHERENCE (cheap, output-side): top-10 mass of decoder->logits
with torch.no_grad():
    WU=model.W_U.detach()                     # [d_model, vocab]
    LE=(Wd.T @ WU)                            # [N, vocab] direct logit effect
    aLE=LE.abs()
    top10=aLE.topk(10,dim=1).values.sum(1)
    coh=top10/(aLE.sum(1)+1e-9)              # fraction of mass in top-10 tokens

# STABILITY
with torch.no_grad():
    W2=sae2.W_dec.weight
    stab=((Wd/Wd.norm(dim=0)).T@(W2/W2.norm(dim=0))).max(1).values

# text run: interpretability + frequency + sample for causal
firetok=[Counter() for _ in range(N)]; fires=torch.zeros(N); docs=[]
for i in range(60):
    ids=model.to_tokens(data[i]["text"][:700])
    with torch.no_grad():
        base,c=model.run_with_cache(ids,return_type="loss")
        _,sp=sae(c["blocks.6.hook_resid_post"][0])
    if i<12: docs.append((ids,base.item(),sp))
    tid=ids[0].tolist(); rows,cols=torch.where(sp>0)
    for r,k in zip(rows.tolist(),cols.tolist()): firetok[k][tid[r]]+=1
    fires+=(sp>0).float().sum(0)
interp=torch.zeros(N)
for f in range(N):
    cnt=firetok[f]; tot=sum(cnt.values())
    if tot>0: interp[f]=sum(v for _,v in cnt.most_common(3))/tot

import statistics
def pear(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b)); sa=sum((x-ma)**2 for x in a)**.5; sb=sum((y-mb)**2 for y in b)**.5
    return cov/(sa*sb) if sa*sb else 0

alive=torch.where(fires>=10)[0]
A=alive.tolist()
print(f"alive: {len(A)}")
print("\n=== does LOGIT-COHERENCE stand alone, or join a cluster? ===")
print(f"coherence vs STABILITY       = {pear([coh[f].item() for f in A],[stab[f].item() for f in A]):+.3f}")
print(f"coherence vs INTERPRETABILITY= {pear([coh[f].item() for f in A],[interp[f].item() for f in A]):+.3f}")

# causal on a sample
sample=alive[coh[alive].argsort()][torch.linspace(0,len(alive)-1,30).long()].tolist()
ic=[]; cc=[]
for f in sample:
    col=Wd[:,f]; tot=0.0; nf=0
    for ids,base,sp in docs:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[("blocks.6.hook_resid_post",hk)])
        tot+=(abl.item()-base); nf+=int((sp[:,f]>0).sum())
    if nf>0: ic.append(coh[f].item()); cc.append(tot/nf)
print(f"coherence vs CAUSATION       = {pear(ic,cc):+.3f}")
print("\n(reference cluster: interp vs causation was +0.37)")
