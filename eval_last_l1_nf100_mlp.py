import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "last"
nf = 100

def get(split):
    P, y = data[split]
    X = P[pooling].float().numpy()
    return X, y.numpy()

Xtr, ytr = get("train")
Xte, yte = get("test")
Xrt, yrt = get("rt")
Xam, yam = get("am")

mu = Xtr.mean(0)
sd = Xtr.std(0)
sd[sd == 0] = 1.0

def std(X):
    return (X - mu) / sd

Xtr_s = std(Xtr)
Xte_s = std(Xte)
Xrt_s = std(Xrt)
Xam_s = std(Xam)

# l1 selection
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr_s, ytr)
coef = np.abs(sel.coef_.ravel())
cols = np.argsort(coef)[::-1][:nf]

Xtr_c = Xtr_s[:, cols]
Xte_c = Xte_s[:, cols]
Xrt_c = Xrt_s[:, cols]
Xam_c = Xam_s[:, cols]

probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_c, ytr)

indist = f1_score(yte, probe.predict(Xte_c))
rt = f1_score(yrt, probe.predict(Xrt_c))
am = f1_score(yam, probe.predict(Xam_c))
mean_ood = (rt + am) / 2

print(f"indist_f1={indist:.6f}")
print(f"rt_f1={rt:.6f}")
print(f"am_f1={am:.6f}")
print(f"mean_ood_f1={mean_ood:.6f}")
