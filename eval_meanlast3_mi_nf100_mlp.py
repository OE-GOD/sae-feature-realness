import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "meanlast3"
nf = 100

Xtr = data["train"][0][pooling].float().numpy()
ytr = data["train"][1].numpy()

mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0

def std(X):
    return (X - mu) / sd

Xtr_s = std(Xtr)

# MI selection
mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]

probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_sel, ytr)

def eval_split(name):
    X = data[name][0][pooling].float().numpy()
    y = data[name][1].numpy()
    Xs = std(X)[:, cols]
    pred = probe.predict(Xs)
    return f1_score(y, pred)

indist = eval_split("test")
rt = eval_split("rt")
am = eval_split("am")
mean_ood = (rt + am) / 2

print(f"indist_f1={indist}")
print(f"rt_f1={rt}")
print(f"am_f1={am}")
print(f"mean_ood_f1={mean_ood}")
