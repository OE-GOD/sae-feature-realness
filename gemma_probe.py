"""Industrial certification, done right: a SPARSE PROBE over SAE features
(not a single feature, since properties split across features). Oracle-free (task
labels), deployment-relevant (fixed released SAE), certified by HELD-OUT F1."""
import os, torch, string
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
data=load_dataset("NeelNanda/pile-10k",split="train")

def lab(tok):
    t=tok.strip()
    return {"digit":t.isdigit() and t!="", "newline":"\n" in tok,
            "punct":t in string.punctuation,
            "cap_word":len(t)>1 and t[0].isupper() and t[1:].islower(),
            "space_pre":tok.startswith(" ")}
keys=list(lab("x"))
acts=[]; labels={k:[] for k in keys}
for i in range(80):
    ids=model.to_tokens(data[i]["text"][:500])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        f=sae.encode(c[hook][0]).float().cpu()
    acts.append(f)
    for p in ids[0].tolist():
        L=lab(model.tokenizer.decode([p]))
        for k in keys: labels[k].append(1.0 if L[k] else 0.0)
    if dev=="mps": torch.mps.empty_cache()
A=torch.cat(acts); D=A.shape[1]; Tn=A.shape[0]; ntr=int(Tn*0.7)
Atr,Ate=A[:ntr],A[ntr:]
print(f"Gemma Scope 16k | tokens {Tn} | sparse PROBE (top-30 feats + logistic reg)\n")

def f1(pred,lab):
    tp=(pred*lab).sum(); fp=(pred*(1-lab)).sum(); fn=((1-pred)*lab).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return (2*p*r/(p+r+1e-9)).item()

print(f"{'TASK':10} {'#feats':>6} {'HELD-OUT F1':>12}  VERDICT")
cert=0
for k in keys:
    y=torch.tensor(labels[k]); ytr,yte=y[:ntr],y[ntr:]
    # select top-30 features by correlation with label on TRAIN
    corr=torch.tensor([((Atr[:,j]-Atr[:,j].mean())*(ytr-ytr.mean())).mean()/(Atr[:,j].std()+1e-9) for j in range(D)])
    sel=corr.abs().topk(30).indices
    Xtr=Atr[:,sel]; Xte=Ate[:,sel]
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    w=torch.zeros(30,requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],0.05)
    for _ in range(300):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(Xtr@w+b,ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred=((Xte@w+b)>0).float()
    h=f1(pred,yte); ok=h>0.8
    if ok: cert+=1
    print(f"{k:10} {30:>6} {h:>12.2f}  {'*** CERTIFIED for industrial use ***' if ok else 'not reliable'}")
print(f"\n{cert}/{len(keys)} feature-based detectors CERTIFIED (held-out F1>0.8, deployed Gemma Scope SAE)")
