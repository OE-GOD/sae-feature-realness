import torch
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "max"; nf = 400

def X(split): return data[split][0][pooling].float().numpy()
def y(split): return data[split][1].numpy()

Xtr, ytr = X("train"), y("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# selection: mutual information
mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]
probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_sel, ytr)

def evalsplit(split):
    Xs = (X(split) - mu) / sd
    return f1_score(y(split), probe.predict(Xs[:, cols]))

indist = evalsplit("test")
rt = evalsplit("rt")
am = evalsplit("am")
mean_ood = (rt + am) / 2
print(f"RESULT indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
