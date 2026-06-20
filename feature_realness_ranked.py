"""Fixed realness tool: RANK features by a continuous composite (no median artifact),
then CAUSAL spot-check the top — does the model actually USE the proxy-'real' features?"""
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

# cheap axes
with torch.no_grad():
    W2=sae2.W_dec.weight
    stab=((Wd/Wd.norm(dim=0)).T@(W2/W2.norm(dim=0))).max(1).values
    LE=(Wd.T@model.W_U.detach()).abs(); coh=LE.topk(10,1).values.sum(1)/(LE.sum(1)+1e-9)

# text pass: interpretability + frequency + docs for ablation
firetok=[Counter() for _ in range(N)]; fires=torch.zeros(N); docs=[]
for i in range(60):
    ids=model.to_tokens(data[i]["text"][:600])
    with torch.no_grad():
        base,c=model.run_with_cache(ids,return_type="loss")
        _,sp=sae(c["blocks.6.hook_resid_post"][0])
    if i<10: docs.append((ids,base.item(),sp))
    tid=ids[0].tolist(); r,cc=torch.where(sp>0)
    for rr,k in zip(r.tolist(),cc.tolist()): firetok[k][tid[rr]]+=1
    fires+=(sp>0).float().sum(0)
interp=torch.zeros(N)
for f in range(N):
    cnt=firetok[f]; t=sum(cnt.values())
    if t>0: interp[f]=sum(v for _,v in cnt.most_common(3))/t

alive=torch.where(fires>=10)[0]
def pctrank(x, idx):  # percentile rank among alive
    v=x[idx]; r=v.argsort().argsort().float()/(len(idx)-1); out=torch.zeros(N); out[idx]=r; return out
# FIX 1: rank-normalized composite (no median threshold artifact)
comp=(pctrank(stab,alive)+pctrank(interp,alive)+pctrank(coh,alive))/3
order=alive[comp[alive].argsort(descending=True)]

# FIX 2: causal spot-check top-20 + 10 random controls
def causal(f):
    col=Wd[:,f]; tot=0.0; nf=0
    for ids,base,sp in docs:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[("blocks.6.hook_resid_post",hk)])
        tot+=abs(abl.item()-base); nf+=int((sp[:,f]>0).sum())
    return tot/nf if nf else 0.0
torch.manual_seed(0)
controls=alive[torch.randperm(len(alive))[:10]].tolist()
cnoise=sorted(causal(f) for f in controls); band=cnoise[len(cnoise)//2]*2  # 2x median control
top=order[:20].tolist()

def toks(f,n=3): return [repr(model.tokenizer.decode([t])) for t,_ in firetok[f].most_common(n)]
print(f"alive features: {len(alive)}   causal noise band (2x median control): {band:.5f}\n")
print(f"{'rank':>4} {'feat':>5} {'comp':>5} {'stab':>5} {'intp':>5} {'caus':>8} {'USED?':>6}  top tokens")
confirmed=0
for i,f in enumerate(top):
    cv=causal(f); used=cv>band; confirmed+=used
    print(f"{i+1:>4} {f:>5} {comp[f]:.2f} {stab[f]:>5.2f} {interp[f]:>5.2f} {cv:>8.4f} {'YES' if used else 'no':>6}  {toks(f)}")
print(f"\nof top-20 proxy-'real' features, CAUSALLY CONFIRMED (ablation > noise): {confirmed}/20")
print("=> this is the real triage tool: ranked by cheap proxies, top verified causally.")
