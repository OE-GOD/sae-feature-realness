import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "max"
nf = 400

def getX(split):
    return data[split][0][pooling].float().numpy()
def gety(split):
    return data[split][1].numpy()

Xtr = getX("train"); ytr = gety("train")

mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0

def std(X):
    return (X - mu) / sd

Xtr_s = std(Xtr)

# mutual info selection
mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_sel, ytr)

def evalsplit(split):
    X = std(getX(split))[:, cols]
    y = gety(split)
    pred = probe.predict(X)
    return f1_score(y, pred)

indist = evalsplit("test")
rt = evalsplit("rt")
am = evalsplit("am")
mean_ood = (rt + am) / 2

print("indist_f1", indist)
print("rt_f1", rt)
print("am_f1", am)
print("mean_ood_f1", mean_ood)
