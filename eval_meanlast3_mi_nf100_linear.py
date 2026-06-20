import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "meanlast3"
nf = 100

Xtr = data["train"][0][pooling].float().numpy()
ytr = data["train"][1].numpy()

mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0

Xtr_s = (Xtr - mu) / sd

# MI selection
mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_sel, ytr)

def eval_split(name):
    X = data[name][0][pooling].float().numpy()
    y = data[name][1].numpy()
    Xs = (X - mu) / sd
    Xsel = Xs[:, cols]
    pred = probe.predict(Xsel)
    return f1_score(y, pred)

indist_f1 = eval_split("test")
rt_f1 = eval_split("rt")
am_f1 = eval_split("am")
mean_ood_f1 = (rt_f1 + am_f1) / 2

print("indist_f1", indist_f1)
print("rt_f1", rt_f1)
print("am_f1", am_f1)
print("mean_ood_f1", mean_ood_f1)
