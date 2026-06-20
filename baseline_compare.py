"""THE baseline test: do SAE features ADD anything, or does a raw-activation probe
match them? newline task on Gemma, in-dist + cross-distribution."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
pile=load_dataset("NeelNanda/pile-10k",split="train")
ood=load_dataset("roneneldan/TinyStories",split="train")

def collect(get,idxs):
    R=[]; S=[]; y=[]
    for i in idxs:
        t=get(i)
        if not t or len(t)<20: continue
        ids=model.to_tokens(t[:500])
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=[hook])
            resid=c[hook][0].float().cpu()         # raw residual [seq,2304]
            feats=sae.encode(c[hook][0]).float().cpu()  # SAE feats [seq,16384]
        R.append(resid); S.append(feats)
        for p in ids[0].tolist(): y.append(1.0 if "\n" in model.tokenizer.decode([p]) else 0.0)
        if dev=="mps": torch.mps.empty_cache()
    return torch.cat(R),torch.cat(S),torch.tensor(y)

Rtr,Str,ytr=collect(lambda i: pile[i]["text"], range(0,60))
Rin,Sin,yin=collect(lambda i: pile[i]["text"], range(60,90))
Rood,Sood,yood=collect(lambda i: ood[i]["text"], range(0,60))
print(f"train {len(ytr)}  in-dist {len(yin)}  OOD {len(yood)}\n")

def lr_fit(X,y,steps=400,wd=0.0):
    mu,sd=X.mean(0),X.std(0)+1e-6; Xn=(X-mu)/sd
    w=torch.zeros(X.shape[1],requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],0.05,weight_decay=wd)
    for _ in range(steps):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(Xn@w+b,y)
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach(),b.detach(),mu,sd
def f1(X,y,w,b,mu,sd,thr):
    p=(((X-mu)/sd)@w+b>thr).float(); tp=(p*y).sum();fp=(p*(1-y)).sum();fn=((1-p)*y).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return (2*pr*rc/(pr+rc+1e-9)).item()
def best_thr(X,y,w,b,mu,sd): return max(torch.linspace(-3,3,40).tolist(),key=lambda t:f1(X,y,w,b,mu,sd,t))

# SAE-feature probe: sparse top-30
corr=torch.tensor([((Str[:,j]-Str[:,j].mean())*(ytr-ytr.mean())).mean()/(Str[:,j].std()+1e-9) for j in range(Str.shape[1])])
sel=corr.abs().topk(30).indices
w,b,mu,sd=lr_fit(Str[:,sel],ytr); thr=best_thr(Str[:,sel],ytr,w,b,mu,sd)
sae_in=f1(Sin[:,sel],yin,w,b,mu,sd,thr); sae_ood=f1(Sood[:,sel],yood,w,b,mu,sd,thr)

# Raw-activation probe: full 2304-dim (strong baseline), light L2
wr,br,mur,sdr=lr_fit(Rtr,ytr,steps=600,wd=1e-3); thrr=best_thr(Rtr,ytr,wr,br,mur,sdr)
raw_in=f1(Rin,yin,wr,br,mur,sdr,thrr); raw_ood=f1(Rood,yood,wr,br,mur,sdr,thrr)

print(f"{'PROBE':28} {'in-dist F1':>10} {'OOD F1':>8}")
print(f"{'SAE-feature (sparse, 30)':28} {sae_in:>10.2f} {sae_ood:>8.2f}")
print(f"{'Raw residual (full, 2304)':28} {raw_in:>10.2f} {raw_ood:>8.2f}")
print(f"\nverdict: SAE features {'ADD capability' if sae_in>raw_in+0.05 else 'MATCH raw (add interpretability/sparsity, not accuracy)' if abs(sae_in-raw_in)<=0.05 else 'UNDERPERFORM raw'}")
