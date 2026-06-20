import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "last"
nf = 100

def get(split):
    P, y = data[split]
    X = P[pooling].float().numpy()
    return X, np.asarray(y)

Xtr, ytr = get("train")
Xte, yte = get("test")
Xrt, yrt = get("rt")
Xam, yam = get("am")

mu = Xtr.mean(0)
sd = Xtr.std(0)
sd[sd == 0] = 1.0

def std(X):
    return (X - mu) / sd

Xtr_s, Xte_s, Xrt_s, Xam_s = std(Xtr), std(Xte), std(Xrt), std(Xam)

mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr_s[:, cols], ytr)

def f1(X, y):
    return f1_score(y, clf.predict(X[:, cols]))

indist = f1(Xte_s, yte)
rt = f1(Xrt_s, yrt)
am = f1(Xam_s, yam)
mean_ood = (rt + am) / 2

print(f"indist_f1={indist}")
print(f"rt_f1={rt}")
print(f"am_f1={am}")
print(f"mean_ood_f1={mean_ood}")
