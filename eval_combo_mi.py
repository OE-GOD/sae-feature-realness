import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "meanlast3", 400

def X(split): return data[split][0][pooling].float().numpy()
def Y(split): return data[split][1].numpy()

Xtr, ytr = X("train"), Y("train")
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
Z = lambda A: (A - mu) / sd
Xtr_s = Z(Xtr)

# mi selection: top-nf by mutual_info_classif
mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(-mi)[:nf]

# mlp probe
probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_s[:, cols], ytr)

def f1(split):
    return f1_score(Y(split), probe.predict(Z(X(split))[:, cols]))

indist = f1("test"); rt = f1("rt"); am = f1("am")
mean_ood = (rt + am) / 2
print("RESULT", indist, rt, am, mean_ood)
