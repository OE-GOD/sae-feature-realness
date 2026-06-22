"""Toward 'what is the model thinking': identify (a) the transfer-stable features the model uses
for ROBUST sentiment, and (b) the fragile features that DRIVE its wrong financial predictions.
Output indices to look up on Neuronpedia (Gemma Scope 12-gemmascope-res-16k)."""
import torch, numpy as np
from idea_harness import H
hs = torch.load("hard_shift_sentiment.pt", weights_only=False)
h = H("max"); w = h.full.coef_[0]; rel = set(h.reliable.tolist())

# (a) top robust sentiment features: high transfer-stability, signed by sentiment direction
c_tr = h.scorr(h.Ztr, h.ytr)
stab = h.feat_stability.copy()                       # corr_tr * corr_rt
top_rel = h.reliable[np.argsort(-stab[h.reliable])[:10]]
print("ROBUST SENTIMENT FEATURES (transfer-stable, what the model reliably uses):")
for j in top_rel:
    sign = "POS" if c_tr[j] > 0 else "NEG"
    print(f"  feat {int(j):5d}  stability={stab[j]:.3f}  sentiment={sign}  probe_w={w[j]:+.2f}")

# (b) fragile features driving WRONG financial predictions
P, y = hs["financial"]; Z = h.std(P["max"].float().numpy()); yy = y.numpy()
m = h.full.decision_function(Z); pred_dir = np.sign(m); wrong = (h.full.predict(Z) != yy)
push = pred_dir[:, None] * (Z * w)                   # per-feature push toward the (wrong) call
alive_unrel = np.array([j for j in h.alive if j not in rel])
mean_push_wrong = push[wrong][:, alive_unrel].mean(0)
top_frag = alive_unrel[np.argsort(-mean_push_wrong)[:10]]
print("\nFRAGILE FEATURES DRIVING WRONG FINANCIAL CALLS (fire spuriously, override the robust core):")
for j in top_frag:
    fires_fin = (Z[:, j] > 0).mean(); fires_tr = (h.Ztr[:, j] > 0).mean()
    print(f"  feat {int(j):5d}  push_on_wrong={push[wrong][:,j].mean():+.2f}  "
          f"fires_financial={fires_fin:.0%} vs fires_train={fires_tr:.0%}  probe_w={w[j]:+.2f}")
print("\nNEURONPEDIA: https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/<feat>")
print("robust:", [int(j) for j in top_rel])
print("fragile:", [int(j) for j in top_frag])
