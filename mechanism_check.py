"""Mechanistic test of WHY reliable-evidence works: on financial (near-chance), are WRONG
predictions driven by UNRELIABLE features while CORRECT ones rest on RELIABLE features?
Partition each prediction's margin into reliable vs unreliable 'push' toward the predicted label."""
import torch, numpy as np
from idea_harness import H
hs = torch.load("hard_shift_sentiment.pt", weights_only=False)
for pool in ["mean","max"]:
    h=H(pool); w=h.full.coef_[0]; rel=set(h.reliable.tolist())
    unrel=np.array([j for j in h.alive if j not in rel]); rela=h.reliable
    for d in ["financial","tweet"]:
        P,y=hs[d]; Z=h.std(P[pool].float().numpy()); yy=y.numpy()
        m=h.full.decision_function(Z); pred_dir=np.sign(m); correct=(h.full.predict(Z)==yy)
        rel_push = pred_dir*(Z[:,rela]@w[rela])     # reliable features' push toward the prediction
        unrel_push = pred_dir*(Z[:,unrel]@w[unrel]) # unreliable features' push toward the prediction
        C=correct; W=~correct
        print(f"[{pool}] {d:9} (base acc {correct.mean():.2f}, n_wrong={W.sum()}):")
        print(f"    CORRECT preds: reliable push {rel_push[C].mean():+.2f}, unreliable push {unrel_push[C].mean():+.2f}")
        print(f"    WRONG   preds: reliable push {rel_push[W].mean():+.2f}, unreliable push {unrel_push[W].mean():+.2f}")
        print(f"    -> reliable push correct-minus-wrong = {rel_push[C].mean()-rel_push[W].mean():+.2f}"
              f"  (positive = wrong preds rest LESS on reliable features, as the thesis predicts)")
