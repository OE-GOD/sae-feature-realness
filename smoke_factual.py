"""Smoke test for the generation/latent-knowledge experiment: can base gemma-2-2b few-shot-JUDGE
True/False factual statements, and can we read (a) its judgment and (b) its internal SAE features
over the statement tokens? If yes, the full experiment (does it 'know' the truth it gets wrong?) is feasible."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
tid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
TRUE,FALSE=tid(" True"),tid(" False")
print("True/False token ids:",TRUE,FALSE)
PREFIX=("Statement: The capital of Italy is Rome.\nAnswer: True\n"
        "Statement: The sun rises in the west.\nAnswer: False\n"
        "Statement: Water is made of hydrogen and oxygen.\nAnswer: True\n"
        "Statement: A triangle has four sides.\nAnswer: False\n")
tests=[("The capital of France is Paris.",1),("The capital of Japan is Beijing.",0),
       ("The chemical symbol for oxygen is O.",1),("The chemical symbol for gold is Xe.",0),
       ("A spider has eight legs.",1),("Three times three equals ten.",0)]
for stmt,truth in tests:
    prompt=PREFIX+f"Statement: {stmt}\nAnswer:"
    toks=model.to_tokens(prompt)
    with torch.no_grad():
        logits,cache=model.run_with_cache(toks,names_filter=[hook])
    lt,lf=logits[0,-1,TRUE].item(),logits[0,-1,FALSE].item()
    judg=1 if lt>lf else 0
    # SAE features over the statement tokens (last ~len(stmt) tokens before "\nAnswer:")
    stmt_toks=model.to_tokens(" "+stmt,prepend_bos=False).shape[1]
    feats=sae.encode(cache[hook][0]).float()
    span=feats[-(stmt_toks+3):-2]  # rough statement span
    print(f"truth={truth} model_judge={judg} {'OK' if judg==truth else 'WRONG'}  "
          f"(Tlogit {lt:.1f} vs Flogit {lf:.1f})  feats_span={tuple(span.shape)}  nonzero={int((span.mean(0)>0).sum())}")
