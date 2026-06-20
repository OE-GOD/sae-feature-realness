import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "last"; nf = 100

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
Xte, yte = get("test")
Xrt, yrt = get("rt")
Xam, yam = get("am")

mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
def std(X): return (X - mu) / sd
Xtr_s = std(Xtr); Xte_s = std(Xte); Xrt_s = std(Xrt); Xam_s = std(Xam)

# l1 selection
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr_s, ytr)
coef = np.abs(sel.coef_[0])
cols = np.argsort(coef)[::-1][:nf]

# probe linear
probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def f1(X, y): return f1_score(y, probe.predict(X[:, cols]))
indist = f1(Xte_s, yte)
rt = f1(Xrt_s, yrt)
am = f1(Xam_s, yam)
mean_ood = (rt + am) / 2
print(f"indist_f1={indist}")
print(f"rt_f1={rt}")
print(f"am_f1={am}")
print(f"mean_ood_f1={mean_ood}")
