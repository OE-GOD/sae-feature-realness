"""Two more operator types, behavioral (clean). DOUBLE-NEGATION: does 'not' reapply as a shift
(not not good -> MORE negative, fails to cancel) or a reflection (returns to positive)? Litotes
'not bad' tested too. CONJUNCTION: is judge(A and B) ~ average of judge(A),judge(B)?"""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item(); POS,NEG=sid(" positive"),sid(" negative")
FEW=("Review: A wonderful, heartwarming film.\nSentiment: positive\nReview: A boring waste of time.\nSentiment: negative\n")
def judge(phrase):
    with torch.no_grad(): lg=model(model.to_tokens(FEW+f"Review: The movie was {phrase}.\nSentiment:"))
    return (lg[0,-1,POS]-lg[0,-1,NEG]).item()

print("=== DOUBLE NEGATION (shift => 'not not good' stays/more negative; reflection => returns +) ===")
print(f"{'adj':>10}{'X':>8}{'not X':>8}{'not not X':>11}")
d1=[];d2=[]
for a in ["good","great","wonderful","brilliant","lovely"]:
    b=judge(a); n=judge(f"not {a}"); nn=judge(f"not not {a}"); d1.append(n-b); d2.append(nn-n)
    print(f"{a:>10}{b:>8.2f}{n:>8.2f}{nn:>11.2f}")
print(f"  mean shift: first 'not' {np.mean(d1):+.2f}, second 'not' {np.mean(d2):+.2f}  (same sign => repeated subtraction, no cancel)")
print(f"  litotes: 'not bad'={judge('not bad'):.2f} vs 'bad'={judge('bad'):.2f} vs 'good'={judge('good'):.2f}  (human 'not bad'~mild +)")

print("\n=== CONJUNCTION (is judge(A and B) ~ average of the parts?) ===")
pairs=[("good","wonderful","pos+pos"),("great","brilliant","pos+pos"),("bad","boring","neg+neg"),
       ("terrible","awful","neg+neg"),("good","boring","MIXED"),("wonderful","terrible","MIXED"),("great","bad","MIXED")]
print(f"{'A':>10}{'B':>10}{'type':>9}{'jA':>7}{'jB':>7}{'A and B':>9}{'avg':>7}")
for A,B,t in pairs:
    jA=judge(A); jB=judge(B); jc=judge(f"{A} and {B}"); print(f"{A:>10}{B:>10}{t:>9}{jA:>7.2f}{jB:>7.2f}{jc:>9.2f}{(jA+jB)/2:>7.2f}")
print("  judge(A and B) ~ avg => compositional averaging; ~max/one-dominates => not simple sum")
