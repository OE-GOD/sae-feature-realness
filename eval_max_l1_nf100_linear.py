import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "max", 100

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
Xte, yte = get("test")
Xrt, yrt = get("rt")
Xam, yam = get("am")

mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
def std(X): return (X - mu) / sd
Xtr, Xte, Xrt, Xam = std(Xtr), std(Xte), std(Xrt), std(Xam)

# l1 selection
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr, ytr)
coef = np.abs(sel.coef_[0])
cols = np.argsort(coef)[::-1][:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr[:, cols], ytr)

indist = f1_score(yte, probe.predict(Xte[:, cols]))
rt = f1_score(yrt, probe.predict(Xrt[:, cols]))
am = f1_score(yam, probe.predict(Xam[:, cols]))
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
