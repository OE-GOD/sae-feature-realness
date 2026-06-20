"""Industrial certification on the REAL, DEPLOYABLE Gemma Scope SAE.
Criterion (oracle-free, deployment-relevant): does a feature reliably detect its
concept on HELD-OUT data? You ship the released SAE as-is, so held-out F1 IS the cert."""
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
firing=[]; labels={k:[] for k in keys}
for i in range(80):
    ids=model.to_tokens(data[i]["text"][:500])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        f=sae.encode(c[hook][0]).float().cpu()
    firing.append((f>0).float())
    for p in ids[0].tolist():
        L=lab(model.tokenizer.decode([p]))
        for k in keys: labels[k].append(1.0 if L[k] else 0.0)
    if dev=="mps": torch.mps.empty_cache()
F=torch.cat(firing); D=sae.cfg.d_sae; Tn=F.shape[0]; ntr=int(Tn*0.7)
print(f"Gemma Scope 16k SAE | tokens {Tn}, train {ntr}/test {Tn-ntr}\n")

def f1(pred,lab):
    tp=(pred*lab).sum(); fp=(pred*(1-lab)).sum(); fn=((1-pred)*lab).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return (2*p*r/(p+r+1e-9)).item(),p.item(),r.item()

print(f"{'TASK':10} {'feat':>6} {'trainF1':>8} {'HELD-OUT F1':>12} {'prec':>5} {'rec':>5}  VERDICT")
cert=0
for k in keys:
    y=torch.tensor(labels[k]); ytr,yte=y[:ntr],y[ntr:]; Ftr,Fte=F[:ntr],F[ntr:]
    f1s=torch.tensor([f1(Ftr[:,j],ytr)[0] for j in range(D)])
    bf=f1s.argmax().item()
    h,p,r=f1(Fte[:,bf],yte)
    ok=h>0.8
    if ok: cert+=1
    print(f"{k:10} {bf:>6} {f1s[bf]:>8.2f} {h:>12.2f} {p:>5.2f} {r:>5.2f}  {'*** CERTIFIED for use ***' if ok else 'not reliable'}")
print(f"\n{cert}/{len(keys)} tasks have a feature CERTIFIED real for industrial use (held-out F1>0.8 on deployed SAE)")
