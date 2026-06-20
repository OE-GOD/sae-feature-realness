import torch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "last"
nf = 400

Xtr = data["train"][0][pooling].float().numpy()
ytr = data["train"][1].numpy()

# Standardize with train mean/std
mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# corr selection: top-nf by |point-biserial corr| with ytr
yc = (ytr - ytr.mean())
denom = (Xtr_s.std(axis=0) * yc.std())
denom[denom == 0] = 1.0
corr = (Xtr_s * yc[:, None]).mean(axis=0) / denom
cols = np.argsort(-np.abs(corr))[:nf]

Xtr_sel = Xtr_s[:, cols]

probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
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

print(f"indist_f1={indist_f1}")
print(f"rt_f1={rt_f1}")
print(f"am_f1={am_f1}")
print(f"mean_ood_f1={mean_ood_f1}")
