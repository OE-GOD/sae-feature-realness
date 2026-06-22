"""Is factual truth decodable AT ALL (in-topic), or genuinely absent? For each topic, train a probe
on half (with truth), test on the other half; key: on the model's OWN ERRORS in the test half, does
the probe recover truth? If yes -> latent knowledge exists but is topic-specific (doesn't transfer).
If no -> the model holds genuine false beliefs (no recoverable truth)."""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
d=torch.load("factual_latent.pt", weights_only=False)
rng=np.random.RandomState(0)
for pool in ["max","mean"]:
    print(f"\n[{pool}] {'topic':11}{'within_acc':>11}{'recover_err_intopic':>21}{'#test_err':>10}")
    for t in d:
        X=d[t][pool].float().numpy(); y=d[t]["truth"]; j=d[t]["judge"]
        idx=rng.permutation(len(y)); h=len(y)//2; tr,te=idx[:h],idx[h:]
        mu,sd=X[tr].mean(0),X[tr].std(0)+1e-6
        clf=LogisticRegression(max_iter=1000,C=0.3).fit((X[tr]-mu)/sd,y[tr])
        pred=clf.predict((X[te]-mu)/sd); acc=(pred==y[te]).mean()
        err=(j[te]!=y[te])
        rec=(pred[err]==y[te][err]).mean() if err.sum() else float('nan')
        print(f"     {t:11}{acc:>11.3f}{rec:>21.1%}{int(err.sum()):>10}")
