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

with torch.no_grad():
    WU=model.W_U.detach(); LE=(Wd.T@WU).abs()
    logit_coh=LE.topk(10,1).values.sum(1)/(LE.sum(1)+1e-9)
    W2=sae2.W_dec.weight
    stability=((Wd/Wd.norm(dim=0)).T@(W2/W2.norm(dim=0))).max(1).values
    dec_norm=Wd.norm(dim=0)

firetok=[Counter() for _ in range(N)]; fires=torch.zeros(N); docs=[]
for i in range(80):
    ids=model.to_tokens(data[i]["text"][:800])
    with torch.no_grad():
        base,c=model.run_with_cache(ids,return_type="loss")
        _,sp=sae(c["blocks.6.hook_resid_post"][0])
    if i<10: docs.append((ids,base.item(),sp))
    tid=ids[0].tolist(); r,cc=torch.where(sp>0)
    for rr,k in zip(r.tolist(),cc.tolist()): firetok[k][tid[rr]]+=1
    fires+=(sp>0).float().sum(0)
interp=torch.zeros(N); toptok=[None]*N
for f in range(N):
    cnt=firetok[f]; tot=sum(cnt.values())
    if tot>0: interp[f]=sum(v for _,v in cnt.most_common(3))/tot; toptok[f]=cnt.most_common(1)[0][0]

import statistics
def pear(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b)); sa=sum((x-ma)**2 for x in a)**.5; sb=sum((y-mb)**2 for y in b)**.5
    return cov/(sa*sb) if sa*sb else 0

alive=torch.where(fires>=15)[0]; A=alive.tolist()
cheap={"freq":[torch.log10(fires[f]).item() for f in A],"stability":[stability[f].item() for f in A],
 "interp":[interp[f].item() for f in A],"logit_coh":[logit_coh[f].item() for f in A],
 "dec_norm":[dec_norm[f].item() for f in A]}
ck=list(cheap)
print(f"CHEAP axes, ALL {len(A)} alive features (low noise):")
print("           "+" ".join(f"{k[:8]:>8}" for k in ck))
for k1 in ck: print(f"{k1[:10]:10} "+" ".join(f"{pear(cheap[k1],cheap[k2]):+7.2f}" for k2 in ck))

# expensive on n=80 sample
sample=alive[torch.linspace(0,len(alive)-1,80).long()].tolist()
caus={}; suff={}
for f in sample:
    col=Wd[:,f]; t=toptok[f]; tot=0.0; nf=0; dl=0.0; nd=0
    for ids,base,sp in docs:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[("blocks.6.hook_resid_post",hk)])
        tot+=(abl.item()-base); nf+=int((sp[:,f]>0).sum())
        def hk2(rp,hook,c=col): rp[0,-1]=rp[0,-1]+4.0*c.to(rp.device); return rp
        with torch.no_grad():
            l0=model(ids)[0,-1]; l1=model.run_with_hooks(ids,fwd_hooks=[("blocks.6.hook_resid_post",hk2)])[0,-1]
        dl+=(l1[t]-l0[t]).item(); nd+=1
    if nf>0: caus[f]=tot/nf; suff[f]=dl/nd
S=[f for f in sample if f in caus]
allv={**{k:[cheap[k][A.index(f)] for f in S] for k in ck},"causation":[caus[f] for f in S],"suffic":[suff[f] for f in S]}
ak=list(allv)
print(f"\nALL 7 axes, n={len(S)} sample:")
print("           "+" ".join(f"{k[:8]:>8}" for k in ak))
for k1 in ak: print(f"{k1[:10]:10} "+" ".join(f"{pear(allv[k1],allv[k2]):+7.2f}" for k2 in ak))
print("\nOVERLAP (|r|>=0.4) = measuring similar things:")
for i,k1 in enumerate(ak):
    for k2 in ak[i+1:]:
        r=pear(allv[k1],allv[k2])
        if abs(r)>=0.4: print(f"  {k1} <-> {k2}: {r:+.2f}")
