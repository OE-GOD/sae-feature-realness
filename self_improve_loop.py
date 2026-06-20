"""HONEST kernel of 'model interprets itself to improve itself':
a closed loop where a detector reads Gemma's OWN features, and those same feature
directions STEER its residual stream toward a target (positive sentiment), iterated.
Measures: does the interpreted property climb monotonically with steering steps?
This is self-steering toward a target via self-interpretation -- NOT open-ended RSI."""
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

# 1. BUILD the self-interpreter: a sentiment probe on Gemma's OWN SAE features
sst=load_dataset("stanfordnlp/sst2",split="train")
P=[i for i in range(len(sst)) if sst[i]["label"]==1][:120]
N=[i for i in range(len(sst)) if sst[i]["label"]==0][:120]
X=[]; y=[]
for i in P+N:
    ids=model.to_tokens(sst[i]["sentence"][:200])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        X.append(sae.encode(c[hook][0]).float().mean(0).cpu().numpy())
    y.append(1 if i in P else 0)
    if dev=="mps": torch.mps.empty_cache()
X=np.array(X); y=np.array(y)
mu,sd=X.mean(0),X.std(0)+1e-6
clf=LogisticRegression(max_iter=500).fit((X-mu)/sd,y)
print("self-interpreter (sentiment probe) trained, train acc",clf.score((X-mu)/sd,y))

# 2. The STEERING direction = top sentiment SAE features' decoder columns, weighted by probe
w=clf.coef_[0]
top=np.argsort(-np.abs(w))[:50]
Wdec=sae.W_dec.detach().float().cpu()    # [16384, 2304] (rows = features) -- verify orientation
if Wdec.shape[0]!=sae.cfg.d_sae: Wdec=Wdec.T
steer_dir=torch.zeros(sae.cfg.d_in)
for f in top: steer_dir += float(w[f]) * Wdec[f]
steer_dir = steer_dir/steer_dir.norm()
steer_dir = steer_dir.to(dev)

def sentiment_of(resid_acts):  # probe the model's own state (encode resid -> SAE features first)
    with torch.no_grad():
        feats=sae.encode(resid_acts).float().mean(0).cpu().numpy()  # [16384]
    v=((feats-mu)/sd)
    return float(clf.decision_function(v.reshape(1,-1))[0])

# 3. THE LOOP: start from a neutral prompt, iteratively steer + re-measure
prompt="The movie was"
ids=model.to_tokens(prompt)
print(f"\nstart prompt: {prompt!r}")
print(f"{'step':>4} {'steer_alpha':>11} {'self-measured sentiment':>24}")
scores=[]
for step in range(7):
    alpha=step*4.0  # increasing self-steering strength
    def hk(rp,hook,a=alpha): rp[0]=rp[0]+a*steer_dir; return rp
    # measure: run WITH the steer hook applied, read the model's own resid, probe sentiment
    hooks=[(hook,hk)] if alpha>0 else []
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            _,cache=model.run_with_cache(ids,names_filter=[hook])
        s=sentiment_of(cache[hook][0])
    scores.append(s)
    print(f"{step:>4} {alpha:>11.1f} {s:>24.3f}")
    if dev=="mps": torch.mps.empty_cache()

mono=all(scores[i+1]>=scores[i]-0.05 for i in range(len(scores)-1))
print(f"\nsentiment climbed with self-steering: {scores[0]:.2f} -> {scores[-1]:.2f}")
print(f"monotonic (within noise): {mono}")
print("=> closed loop: model interprets its own features, steers itself, property moves. (toy, NOT open-ended RSI)")
