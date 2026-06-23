"""Prototype DRIVER DETECTOR: for the top READABLE sentiment SAE features, measure each one's
CAUSAL effect (steer along its decoder direction, watch the sentiment output move). Question:
does read-score predict cause-score, or do readable features split into drivers and thermometers?
If read doesn't predict cause, you MUST measure causation to know which features the model thinks with."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
L=12; HOOK=f"blocks.{L}.hook_resid_post"; sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
POS,NEG=sid(" positive"),sid(" negative")
FEW=("Review: A wonderful, heartwarming film.\nSentiment: positive\n"
     "Review: A boring, pointless waste of time.\nSentiment: negative\n")
def readout(text, steer=None):
    toks=model.to_tokens(FEW+f"Review: {text}\nSentiment:")
    hooks=[(HOOK, lambda r,hook: r+steer)] if steer is not None else []
    with torch.no_grad(): lg=model.run_with_hooks(toks,fwd_hooks=hooks)
    return (lg[0,-1,POS]-lg[0,-1,NEG]).item()

# read-score per SAE feature from precollected sentiment features (max pool)
d=torch.load("sem_pooling_dataset.pt", weights_only=False)
X=d["train"][0]["max"].float().numpy(); y=d["train"][1].numpy()
yc=y-y.mean(); zc=X-X.mean(0); read=np.nan_to_num((zc*yc[:,None]).mean(0)/(zc.std(0)*yc.std()+1e-9))
freq=(X>0).mean(0); alive=np.where(freq>0.02)[0]
top=alive[np.argsort(-np.abs(read[alive]))[:20]]                 # top-20 READABLE sentiment features
Wdec=sae.W_dec.detach().float()                                  # [16384, 2304] decoder dirs
ds=load_dataset("stanfordnlp/sst2",split="train"); test=[ds[i]["sentence"] for i in range(400) if ds[i]["label"]==1][:6]+[ds[i]["sentence"] for i in range(400) if ds[i]["label"]==0][:6]
base=np.mean([readout(t) for t in test])
print(f"baseline pos-neg logit={base:.2f}\n{'feat':>7}{'read':>8}{'cause(steer+)':>15}{'verdict':>12}")
rows=[]
for f in top:
    dirv=Wdec[f]; dirv=dirv/dirv.norm()*13.0                     # steer +1 class-magnitude along feature dir, signed by read sign
    sd=(np.sign(read[f])*dirv).to(model.cfg.dtype)
    cause=np.mean([readout(t,steer=sd) for t in test])-base       # change toward positive when pushing the feature's positive direction
    rows.append((int(f),read[f],cause)); 
    print(f"{int(f):>7}{read[f]:>8.3f}{cause:>15.2f}{'DRIVER' if abs(cause)>0.5 else 'thermometer':>12}")
r=np.array([(x[1],x[2]) for x in rows])
print(f"\ncorr(|read|, |cause|) across top features = {np.corrcoef(np.abs(r[:,0]),np.abs(r[:,1]))[0,1]:+.2f}")
print("If low/zero: read-score does NOT predict cause-score -> you must MEASURE causation (steer/AtP) to find the levers.")
