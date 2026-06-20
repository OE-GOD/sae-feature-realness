import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "last"
nf = 400

def get_X(split):
    return data[split][0][pooling].float().numpy()

def get_y(split):
    return data[split][1].numpy()

Xtr = get_X("train"); ytr = get_y("train")
Xte = get_X("test"); yte = get_y("test")
Xrt = get_X("rt"); yrt = get_y("rt")
Xam = get_X("am"); yam = get_y("am")

mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0

def std(X):
    return (X - mu) / sd

Xtr_s = std(Xtr)
Xte_s = std(Xte)
Xrt_s = std(Xrt)
Xam_s = std(Xam)

# MI selection
mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]
Xte_sel = Xte_s[:, cols]
Xrt_sel = Xrt_s[:, cols]
Xam_sel = Xam_s[:, cols]

probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_sel, ytr)

indist_f1 = f1_score(yte, probe.predict(Xte_sel))
rt_f1 = f1_score(yrt, probe.predict(Xrt_sel))
am_f1 = f1_score(yam, probe.predict(Xam_sel))
mean_ood = (rt_f1 + am_f1) / 2

print(f"INDIST {indist_f1}")
print(f"RT {rt_f1}")
print(f"AM {am_f1}")
print(f"MEANOOD {mean_ood}")
