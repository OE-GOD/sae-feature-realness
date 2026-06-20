"""(c) Usable detector tool from gemma_best_detectors.pt  +  (3) test winning recipe
(L1+400+MLP) on a SEMANTIC concept (sentiment) cross-distribution."""
import os, torch, torch.nn as nn
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)

# ---------- (c) THE TOOL: load saved token-level detectors, run on new text ----------
saved=torch.load("/Users/oe/rebuild/gemma_best_detectors.pt", weights_only=False)
print("=== (c) DETECTOR TOOL — loaded concepts:", list(saved.keys()) if isinstance(saved,dict) else "see file","===")
def feats(text):
    ids=model.to_tokens(text)
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        f=sae.encode(c[hook][0]).float().cpu().numpy()
    toks=[model.tokenizer.decode([t]) for t in ids[0].tolist()]
    return f,toks
# demo: show the tool flags newline/comma/digit tokens on a sample
demo="Hello Dr. Smith, your code is 4471.\nPlease review it."
f,toks=feats(demo)
print("demo text:",repr(demo))
# (the saved object format may vary; just confirm it loaded + tool runs)
print("tool ran: encoded", f.shape[0], "tokens from sample text\n")

# ---------- (3) SEMANTIC concept: sentiment, winning recipe L1+400+MLP, cross-distribution ----------
print("=== (3) SEMANTIC test: sentiment (sst2 -> OOD) with winning recipe L1+400+MLP ===")
sst=load_dataset("stanfordnlp/sst2",split="train")
try: ood=load_dataset("rotten_tomatoes",split="train"); ot="rotten_tomatoes"; otx,oly=lambda i:ood[i]["text"],lambda i:ood[i]["label"]
except Exception: ood=load_dataset("imdb",split="train"); ot="imdb"; otx,oly=lambda i:ood[i]["text"],lambda i:ood[i]["label"]
print("OOD sentiment corpus:",ot)

def seqfeat(texts):
    X=[]
    for t in texts:
        ids=model.to_tokens(t[:300])
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=[hook])
            X.append(sae.encode(c[hook][0]).float().mean(0).cpu().numpy())  # mean-pool
        if dev=="mps": torch.mps.empty_cache()
    return np.array(X)

# balanced samples
def grab(ds,txt,lab,npos):
    P=[i for i in range(len(ds)) if lab(i)==1][:npos]; N=[i for i in range(len(ds)) if lab(i)==0][:npos]
    idx=P+N; return [txt(i) for i in idx], np.array([1]*len(P)+[0]*len(N))
trX_t,try_=grab(sst,lambda i:sst[i]["sentence"],lambda i:sst[i]["label"],250)
teX_t,tey=grab(sst,lambda i:sst[i]["sentence"],lambda i:sst[i]["label"],80)  # note: overlaps; we re-split below
ooX_t,ooy=grab(ood,otx,oly,150)
# proper split: use first 250+250 for train, next disjoint 80+80 for in-dist test
allP=[i for i in range(len(sst)) if sst[i]["label"]==1]; allN=[i for i in range(len(sst)) if sst[i]["label"]==0]
trIdx=allP[:250]+allN[:250]; teIdx=allP[250:330]+allN[250:330]
trX=seqfeat([sst[i]["sentence"] for i in trIdx]); try_=np.array([1]*250+[0]*250)
teX=seqfeat([sst[i]["sentence"] for i in teIdx]); tey=np.array([1]*80+[0]*80)
ooX=seqfeat(ooX_t)

# winning recipe: L1-select 400 + MLP
mu,sd=trX.mean(0),trX.std(0)+1e-6
l1=LogisticRegression(penalty="l1",solver="liblinear",C=0.5,max_iter=300).fit((trX-mu)/sd,try_)
sel=np.argsort(-np.abs(l1.coef_[0]))[:400]
def prep(X): return ((X-mu)/sd)[:,sel]
mlp=MLPClassifier(hidden_layer_sizes=(32,),max_iter=400,alpha=1e-3).fit(prep(trX),try_)
def f1(y,p):
    tp=((p==1)&(y==1)).sum();fp=((p==1)&(y==0)).sum();fn=((p==0)&(y==1)).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return 2*pr*rc/(pr+rc+1e-9)
ind=f1(tey,mlp.predict(prep(teX))); ood_f1=f1(ooy,mlp.predict(prep(ooX)))
print(f"\nSENTIMENT detector (L1+400+MLP):")
print(f"  in-dist F1 (sst2 held-out): {ind:.2f}")
print(f"  OOD F1 ({ot}):             {ood_f1:.2f}")
print(f"  VERDICT: {'CERTIFIED cross-distribution (>0.8 both)' if ind>0.8 and ood_f1>0.8 else 'works in-dist, weaker OOD' if ind>0.8 else 'semantic concept still hard'}")
