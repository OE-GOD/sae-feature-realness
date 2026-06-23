"""Clean intensification test, no pooling artifact: does the model's JUDGMENT amplify both poles?
judge('very {adj}') vs judge('{adj}') behaviorally (few-shot sentiment logit). True gain => positives
go MORE positive AND negatives MORE negative. If only positives amplify => operation-general asymmetry."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item(); POS,NEG=sid(" positive"),sid(" negative")
FEW=("Review: A wonderful, heartwarming film.\nSentiment: positive\nReview: A boring waste of time.\nSentiment: negative\n")
def judge(text):
    with torch.no_grad(): lg=model(model.to_tokens(FEW+f"Review: {text}\nSentiment:"))
    return (lg[0,-1,POS]-lg[0,-1,NEG]).item()
PADJ=["good","great","brilliant","lovely","superb","delightful"]; NADJ=["bad","terrible","awful","horrible","dreadful","lousy"]
print(f"{'adj':>10}{'bare':>8}{'very':>8}{'extremely':>11}{'very-bare':>11}")
peff=[];neff=[]
for a in PADJ:
    b=judge(f"The movie was {a}."); v=judge(f"The movie was very {a}."); e=judge(f"The movie was extremely {a}.")
    peff.append(v-b); print(f"{a:>10}{b:>8.2f}{v:>8.2f}{e:>11.2f}{v-b:>+11.2f}")
for a in NADJ:
    b=judge(f"The movie was {a}."); v=judge(f"The movie was very {a}."); e=judge(f"The movie was extremely {a}.")
    neff.append(v-b); print(f"{a:>10}{b:>8.2f}{v:>8.2f}{e:>11.2f}{v-b:>+11.2f}")
print(f"\nmean 'very' effect: POSITIVES {np.mean(peff):+.2f} (gain=>should be +), NEGATIVES {np.mean(neff):+.2f} (gain=>should be -)")
print("both poles amplify => intensification symmetric. only positives => operation-general signed-axis asymmetry.")
