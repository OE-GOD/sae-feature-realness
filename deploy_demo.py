"""REAL-USE deployment test: load the SAVED certified detector and run it on
fresh, unseen input — including a code snippet (a domain it was NOT built on)."""
import os, json, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
det=json.load(open("/Users/oe/rebuild/detector_newline.json"))
print(f"DEPLOYED DETECTOR: '{det['concept']}'  ({len(det['feature_ids'])} features)")
print(f"  certified held-out: {det['held_out']}\n")
sel=torch.tensor(det["feature_ids"]); w=torch.tensor(det["weights"]); b=det["bias"]
mu=torch.tensor(det["feat_mean"]); sd=torch.tensor(det["feat_std"]); thr=det["threshold"]

sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)

def run_detector(text):
    ids=model.to_tokens(text)
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        f=sae.encode(c[hook][0]).float().cpu()
    Z=(f[:,sel]-mu)/sd
    pred=((Z@w+b)>thr).float()
    toks=[model.tokenizer.decode([t]) for t in ids[0].tolist()]
    truth=torch.tensor([1.0 if "\n" in t else 0.0 for t in toks])
    return toks,pred,truth

def metrics(pred,truth):
    tp=(pred*truth).sum(); fp=(pred*(1-truth)).sum(); fn=((1-pred)*truth).sum()
    p=(tp/(tp+fp+1e-9)).item(); r=(tp/(tp+fn+1e-9)).item()
    return p,r,(2*p*r/(p+r+1e-9))

# CASE 1: fresh unseen Pile prose
data=load_dataset("NeelNanda/pile-10k",split="train")
toks,pred,truth=run_detector(data[500]["text"][:400])
p,r,f=metrics(pred,truth)
print(f"CASE 1 — fresh Pile prose (doc 500, unseen):  precision {p:.2f} recall {r:.2f} F1 {f:.2f}")

# CASE 2 — CODE (a domain the detector was NOT built on)
code='''def add(a, b):
    result = a + b
    return result

for i in range(10):
    print(add(i, i+1))
'''
toks,pred,truth=run_detector(code)
p,r,f=metrics(pred,truth)
print(f"CASE 2 — Python code (NEW domain):           precision {p:.2f} recall {r:.2f} F1 {f:.2f}")
print("\n  live output (« » = detector flagged this token as newline):")
out=""
for t,pr in zip(toks,pred.tolist()):
    show=t.replace("\n","\\n")
    out+= f"«{show}»" if pr>0.5 else show
print("  "+out[:600])
