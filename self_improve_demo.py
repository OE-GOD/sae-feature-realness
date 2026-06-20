"""STRONG demo: can self-steering FLIP behavior, not just nudge?
Negative/neutral baselines, steer BOTH directions, measure with probe-INDEPENDENT
logit gap + show actual generated text."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)

sst=load_dataset("stanfordnlp/sst2",split="train")
P=[i for i in range(len(sst)) if sst[i]["label"]==1][:120]; N=[i for i in range(len(sst)) if sst[i]["label"]==0][:120]
X=[];y=[]
for i in P+N:
    ids=model.to_tokens(sst[i]["sentence"][:200])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook]); X.append(sae.encode(c[hook][0]).float().mean(0).cpu().numpy())
    y.append(1 if i in P else 0)
    if dev=="mps": torch.mps.empty_cache()
X=np.array(X);y=np.array(y); mu,sd=X.mean(0),X.std(0)+1e-6
clf=LogisticRegression(max_iter=500).fit((X-mu)/sd,y)
w=clf.coef_[0]; top=np.argsort(-np.abs(w))[:50]
Wdec=sae.W_dec.detach().float().cpu()
if Wdec.shape[0]!=sae.cfg.d_sae: Wdec=Wdec.T
sdir=torch.zeros(sae.cfg.d_in)
for f in top: sdir+=float(w[f])*Wdec[f]
sdir=(sdir/sdir.norm()).to(dev)

pos_words=[" great"," wonderful"," amazing"," excellent"," good"," love"," beautiful"]
neg_words=[" terrible"," awful"," horrible"," bad"," boring"," worst"," ugly"]
def tok1(W): return [model.to_tokens(x,prepend_bos=False)[0,0].item() for x in W]
pos_ids,neg_ids=tok1(pos_words),tok1(neg_words)

def gen(ids,alpha,n=14):
    def hk(rp,hook,a=alpha): rp[0]=rp[0]+a*sdir; return rp
    hooks=[(hook,hk)] if alpha!=0 else []
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            lg=model(ids)[0,-1]; gap=(lg[pos_ids].mean()-lg[neg_ids].mean()).item()
            cur=ids.clone()
            for _ in range(n):
                nx=model(cur)[0,-1].argmax().view(1,1); cur=torch.cat([cur,nx],1)
            txt=model.tokenizer.decode(cur[0,ids.shape[1]:].tolist())
    return gap,txt

prompts=["I watched the film and","The restaurant we went to was","Honestly, my day today was"]
for p in prompts:
    ids=model.to_tokens(p)
    print(f"\n=== prompt: {p!r} ===")
    for alpha in [-28.0,0.0,28.0]:
        g,t=gen(ids,alpha)
        tag="STEER NEG" if alpha<0 else "STEER POS" if alpha>0 else "baseline "
        print(f"  {tag} (a={alpha:+.0f})  gap={g:+.2f}  {t!r}")
    if dev=="mps": torch.mps.empty_cache()
print("\nSTRONG demo: if NEG steering makes text negative + lowers gap, and POS steering")
print("makes it positive + raises gap, the self-steering genuinely controls behavior both ways.")
