"""Controlled read-vs-write, same representation. For sentiment and truth, at each layer build the
diff-of-means direction (raw-text residual, mean-pooled). READ = AUROC of held-out projections onto
that direction vs label. WRITE = steering swing (logitdiff at alpha=+2 minus alpha=-2) using the SAME
direction. Driver = read AND write; thermometer = read but not write. Tests if facts are thermometers."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset
from scipy.stats import rankdata
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
LAYERS=[6,9,12,15,18]; HOOKS=[f"blocks.{L}.hook_resid_post" for L in LAYERS]
def auroc(s,y):
    p=s[y==1];n=s[y==0]; r=rankdata(np.concatenate([p,n])); return (r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))

def resid_all(text):  # mean-pooled resid at each layer for raw text
    toks=model.to_tokens(text[:200])
    with torch.no_grad(): _,c=model.run_with_cache(toks,names_filter=HOOKS)
    return {L:c[f"blocks.{L}.hook_resid_post"][0].float().mean(0) for L in LAYERS}
def readout(prompt_text, fmt, posid, negid, few, L=None, steer=None):
    toks=model.to_tokens(few+fmt.format(prompt_text))
    hooks=[(f"blocks.{L}.hook_resid_post", lambda r,hook: r+steer)] if steer is not None else []
    with torch.no_grad(): lg=model.run_with_hooks(toks,fwd_hooks=hooks)
    return (lg[0,-1,posid]-lg[0,-1,negid]).item()

def run_task(name, pos, neg, fmt, posid, negid, few):
    dpos=[resid_all(t) for t in pos[:40]]; dneg=[resid_all(t) for t in neg[:40]]
    test=pos[40:56]+neg[40:56]; ylab=np.array([1]*16+[0]*16)
    dtest=[resid_all(t) for t in test]
    print(f"\n=== {name} ===\n{'layer':6}{'read_AUROC':>12}{'write_swing':>13}")
    for L in LAYERS:
        d=(torch.stack([x[L] for x in dpos]).mean(0)-torch.stack([x[L] for x in dneg]).mean(0))
        proj=np.array([float(x[L]@d) for x in dtest]); rd=auroc(proj,ylab)
        sw=[]
        for t in test:
            sp=readout(t,fmt,posid,negid,few,L=L,steer=(2*d).to(model.cfg.dtype))
            sn=readout(t,fmt,posid,negid,few,L=L,steer=(-2*d).to(model.cfg.dtype))
            sw.append(sp-sn)
        print(f"{L:6}{rd:>12.3f}{np.mean(sw):>13.2f}")

ds=load_dataset("stanfordnlp/sst2",split="train")
sp=[ds[i]["sentence"] for i in range(3000) if ds[i]["label"]==1][:56]; sn=[ds[i]["sentence"] for i in range(3000) if ds[i]["label"]==0][:56]
run_task("SENTIMENT (concept)", sp, sn, "Review: {}\nSentiment:", sid(" positive"), sid(" negative"),
         "Review: A wonderful, heartwarming film.\nSentiment: positive\nReview: A boring waste of time.\nSentiment: negative\n")
fd=load_dataset("notrichardren/azaria-mitchell",split="train")
idx=[i for i in range(len(fd)) if fd[i]["dataset"] in ("cities","companies","inventions")]
tp=[fd[i]["claim"] for i in idx if fd[i]["label"]==1][:56]; tn=[fd[i]["claim"] for i in idx if fd[i]["label"]==0][:56]
run_task("TRUTH (fact)", tp, tn, "Statement: {}\nAnswer:", sid(" True"), sid(" False"),
         "Statement: The capital of Italy is Rome.\nAnswer: True\nStatement: The sun rises in the west.\nAnswer: False\n")
print("\nDriver = high read AND high write. Thermometer = high read, low write. Compare the two tasks.")
