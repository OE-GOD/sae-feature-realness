"""HONESTY CHECK: does self-steering change actual BEHAVIOR, or only the (circular) probe?
Independent measure: generate text under each steering strength + score positivity by
the logit gap of positive vs negative WORDS (not the SAE probe). Plus show the text."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import numpy as np

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)

# rebuild steering dir (same as loop)
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

# INDEPENDENT sentiment signal: logit gap of pos vs neg words at the final position
pos_words=[" great"," wonderful"," amazing"," excellent"," fantastic"," good"," love"]
neg_words=[" terrible"," awful"," horrible"," bad"," boring"," worst"," hate"]
def tok1(wlist): return [model.to_tokens(x,prepend_bos=False)[0,0].item() for x in wlist]
pos_ids,neg_ids=tok1(pos_words),tok1(neg_words)

prompt="The movie was"
ids=model.to_tokens(prompt)
print(f"prompt: {prompt!r}\n")
print(f"{'alpha':>6} {'INDEP pos-neg logit gap':>24}   generated continuation")
for alpha in [0.0,8.0,16.0,24.0,32.0]:
    def hk(rp,hook,a=alpha): rp[0]=rp[0]+a*sdir; return rp
    hooks=[(hook,hk)] if alpha>0 else []
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            logits=model(ids)[0,-1]
            gap=(logits[pos_ids].mean()-logits[neg_ids].mean()).item()
            # greedy generate 12 tokens under steering
            cur=ids.clone()
            for _ in range(12):
                lg=model(cur)[0,-1]; nxt=lg.argmax().view(1,1); cur=torch.cat([cur,nxt],dim=1)
            cont=model.tokenizer.decode(cur[0,ids.shape[1]:].tolist())
    print(f"{alpha:>6.0f} {gap:>24.3f}   {cont!r}")
    if dev=="mps": torch.mps.empty_cache()
print("\nINDEPENDENT check: if the pos-neg logit gap rises AND the text gets more positive,")
print("the steering changed real behavior -- not just the circular probe.")
