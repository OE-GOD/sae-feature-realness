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

# cheap all-feature axes
with torch.no_grad():
    WU=model.W_U.detach(); LE=(Wd.T@WU).abs()
    logit_coh=LE.topk(10,1).values.sum(1)/(LE.sum(1)+1e-9)
    W2=sae2.W_dec.weight
    stability=((Wd/Wd.norm(dim=0)).T@(W2/W2.norm(dim=0))).max(1).values
    dec_norm=Wd.norm(dim=0)

# text run: interp, freq, contexts, top token
firetok=[Counter() for _ in range(N)]; fires=torch.zeros(N); docs=[]
for i in range(40):
    ids=model.to_tokens(data[i]["text"][:700])
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
    if tot>0:
        interp[f]=sum(v for _,v in cnt.most_common(3))/tot
        toptok[f]=cnt.most_common(1)[0][0]

alive=torch.where(fires>=10)[0]
sample=alive[torch.linspace(0,len(alive)-1,40).long()].tolist()

# expensive: causation (ablation) + sufficiency (steering)
causation={}; sufficiency={}
for f in sample:
    col=Wd[:,f]; t=toptok[f]
    tot=0.0; nf=0; dlog=0.0; nd=0
    for ids,base,sp in docs:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[("blocks.6.hook_resid_post",hk)])
        tot+=(abl.item()-base); nf+=int((sp[:,f]>0).sum())
        # steering: ADD 4*col at last pos, measure d logit of its own token t
        def hk2(rp,hook,c=col): rp[0,-1]=rp[0,-1]+4.0*c.to(rp.device); return rp
        with torch.no_grad():
            l0=model(ids)[0,-1]; l1=model.run_with_hooks(ids,fwd_hooks=[("blocks.6.hook_resid_post",hk2)])[0,-1]
        dlog+=(l1[t]-l0[t]).item(); nd+=1
    if nf>0: causation[f]=tot/nf
    sufficiency[f]=dlog/nd

S=[f for f in sample if f in causation]
cols={"freq":[fires[f].item() for f in S],"stability":[stability[f].item() for f in S],
 "interp":[interp[f].item() for f in S],"logit_coh":[logit_coh[f].item() for f in S],
 "dec_norm":[dec_norm[f].item() for f in S],"causation":[causation[f] for f in S],
 "suffic":[sufficiency[f] for f in S]}
import statistics
def pear(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b)); sa=sum((x-ma)**2 for x in a)**.5; sb=sum((y-mb)**2 for y in b)**.5
    return cov/(sa*sb) if sa*sb else 0
keys=list(cols)
print(f"\nfeatures: {len(S)}    CORRELATION MATRIX of realness axes")
print("           "+" ".join(f"{k[:8]:>8}" for k in keys))
for k1 in keys:
    print(f"{k1[:10]:10} "+" ".join(f"{pear(cols[k1],cols[k2]):+7.2f}" for k2 in keys))
print("\nNOT run (need new data/infra): cross-dataset, cross-layer, paraphrase-invariance")
