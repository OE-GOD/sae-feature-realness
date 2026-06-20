"""Reusable detector-validation tool. Point it at a concept (labeling fn);
it builds a sparse feature-probe, validates on a SEPARATE held-out doc set,
and SAVES a deployable certified detector. The 8-step recipe, productized."""
import os, torch, string, json
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
data=load_dataset("NeelNanda/pile-10k",split="train")

# ---- the concept to certify a detector for ----
CONCEPT="newline"
def label_fn(tok): return 1.0 if "\n" in tok else 0.0
BAR=0.80; K=30

def collect(doc_ids):
    acts=[]; labs=[]
    for i in doc_ids:
        ids=model.to_tokens(data[i]["text"][:500])
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=[hook])
            f=sae.encode(c[hook][0]).float().cpu()
        acts.append(f)
        for p in ids[0].tolist(): labs.append(label_fn(model.tokenizer.decode([p])))
        if dev=="mps": torch.mps.empty_cache()
    return torch.cat(acts), torch.tensor(labs)

# step 3: TRAIN and HELD-OUT come from DIFFERENT documents (real generalization test)
Xtr,ytr=collect(range(0,60))
Xte,yte=collect(range(60,95))
print(f"concept='{CONCEPT}'  train tokens {len(ytr)} (docs 0-59)  HELD-OUT {len(yte)} (docs 60-94)\n")

# step 4: select sparse features by train correlation
corr=torch.tensor([((Xtr[:,j]-Xtr[:,j].mean())*(ytr-ytr.mean())).mean()/(Xtr[:,j].std()+1e-9) for j in range(Xtr.shape[1])])
sel=corr.abs().topk(K).indices
mu,sd=Xtr[:,sel].mean(0),Xtr[:,sel].std(0)+1e-6
Ztr=(Xtr[:,sel]-mu)/sd; Zte=(Xte[:,sel]-mu)/sd

# step 5+6: fit logistic regression, tune threshold on train
w=torch.zeros(K,requires_grad=True); b=torch.zeros(1,requires_grad=True)
opt=torch.optim.Adam([w,b],0.05)
for _ in range(400):
    loss=torch.nn.functional.binary_cross_entropy_with_logits(Ztr@w+b,ytr)
    opt.zero_grad(); loss.backward(); opt.step()
def prf(logits,y,thr):
    pred=(logits>thr).float(); tp=(pred*y).sum(); fp=(pred*(1-y)).sum(); fn=((1-pred)*y).sum()
    p=(tp/(tp+fp+1e-9)).item(); r=(tp/(tp+fn+1e-9)).item(); return p,r,(2*p*r/(p+r+1e-9))
with torch.no_grad():
    str_=Ztr@w+b
    best=max(torch.linspace(-3,3,40).tolist(), key=lambda t: prf(str_,ytr,t)[2])
    p,r,f=prf(Zte@w+b,yte,best)            # step 7: HELD-OUT judgment

print(f"HELD-OUT (unseen docs):  precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}")
ok=f>BAR
print("VERDICT:", "*** CERTIFIED — deployable detector ***" if ok else "not reliable")

# step 8: SAVE the deployable detector
det={"concept":CONCEPT,"sae":"gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_82",
     "feature_ids":sel.tolist(),"weights":w.detach().tolist(),"bias":b.item(),
     "feat_mean":mu.tolist(),"feat_std":sd.tolist(),"threshold":best,
     "held_out":{"precision":p,"recall":r,"f1":f},"certified":bool(ok),
     "scope":f"detects '{CONCEPT}' tokens only; held-out on unseen Pile docs"}
json.dump(det,open(f"/Users/oe/rebuild/detector_{CONCEPT}.json","w"),indent=2)
print(f"\nsaved deployable detector -> detector_{CONCEPT}.json ({K} features)")
