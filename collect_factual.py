"""GENERATION/latent-knowledge experiment collection. For factual True/False claims across topics:
base gemma-2-2b few-shot JUDGES each (its own answer), and we extract its internal SAE features
over the claim tokens + the ground truth. Question this enables: when the model's JUDGMENT is wrong,
does its internal state still encode the truth (recoverable by mass-mean on transfer-stable feats)?"""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
tid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
TRUE,FALSE=tid(" True"),tid(" False")
PREFIX=("Statement: The capital of Italy is Rome.\nAnswer: True\n"
        "Statement: The sun rises in the west.\nAnswer: False\n"
        "Statement: Water is made of hydrogen and oxygen.\nAnswer: True\n"
        "Statement: A triangle has four sides.\nAnswer: False\n")
n_pre=model.to_tokens(PREFIX+"Statement:").shape[1]

def judge_and_feats(claim):
    pre_claim=PREFIX+f"Statement:{claim}"; full=pre_claim+"\nAnswer:"
    n_claim_end=model.to_tokens(pre_claim).shape[1]
    toks=model.to_tokens(full)
    with torch.no_grad():
        logits,cache=model.run_with_cache(toks,names_filter=[hook])
        f=sae.encode(cache[hook][0]).float()
    span=f[n_pre:n_claim_end]                       # SAE feats over the claim tokens
    judge=1 if logits[0,-1,TRUE].item()>logits[0,-1,FALSE].item() else 0
    if dev=="mps": torch.mps.empty_cache()
    return judge, span.mean(0).half().cpu(), span.max(0).values.half().cpu()

ds=load_dataset("notrichardren/azaria-mitchell",split="train")
TOPICS=["cities","companies","animals","inventions","elements"]
N=150
out={}
for topic in TOPICS:
    idx=[i for i in range(len(ds)) if ds[i]["dataset"]==topic]
    pos=[i for i in idx if ds[i]["label"]==1][:N//2]; neg=[i for i in idx if ds[i]["label"]==0][:N//2]
    sel=pos+neg
    if not sel: print("skip",topic); continue
    M=[];X=[];J=[];T=[]
    for i in sel:
        j,m,x=judge_and_feats(ds[i]["claim"]); M.append(m);X.append(x);J.append(j);T.append(ds[i]["label"])
    M=torch.stack(M);X=torch.stack(X);J=np.array(J);T=np.array(T)
    out[topic]={"mean":M,"max":X,"judge":J,"truth":T}
    acc=(J==T).mean()
    print(f"{topic}: n={len(T)} model_judge_acc={acc:.3f} (errors={int((J!=T).sum())})")
torch.save(out,"/Users/oe/rebuild/factual_latent.pt")
print("saved factual_latent.pt")