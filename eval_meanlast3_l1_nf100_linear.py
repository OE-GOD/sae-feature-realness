import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "meanlast3"
nf = 100

Xtr = data["train"][0][pooling].float().numpy()
ytr = data["train"][1].numpy()

# standardize with train mean/std
mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# l1 selection
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr_s, ytr)
coef = np.abs(sel.coef_[0])
cols = np.argsort(coef)[::-1][:nf]

# probe = linear
probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def evalsplit(name):
    X = data[name][0][pooling].float().numpy()
    y = data[name][1].numpy()
    Xs = (X - mu) / sd
    pred = probe.predict(Xs[:, cols])
    return f1_score(y, pred)

indist = evalsplit("test")
rt = evalsplit("rt")
am = evalsplit("am")
mean_ood = (rt + am) / 2

print(f"indist_f1={indist}")
print(f"rt_f1={rt}")
print(f"am_f1={am}")
print(f"mean_ood_f1={mean_ood}")
