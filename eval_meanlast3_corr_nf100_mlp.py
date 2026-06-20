import torch, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "meanlast3", 100

def X(split): return data[split][0][pooling].float().numpy()
def Y(split): return data[split][1].numpy()

Xtr, ytr = X("train"), Y("train")
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
Z = lambda A: (A - mu) / sd
Xtr_s = Z(Xtr)

# corr selection: top-nf by |point-biserial corr| with ytr
yc = ytr - ytr.mean()
yc = yc / (np.linalg.norm(yc) + 1e-12)
Xc = Xtr_s - Xtr_s.mean(0)
Xc = Xc / (np.linalg.norm(Xc, axis=0) + 1e-12)
corr = np.abs(Xc.T @ yc)
cols = np.argsort(-corr)[:nf]

# mlp probe
probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_s[:, cols], ytr)

def f1(split):
    return f1_score(Y(split), probe.predict(Z(X(split))[:, cols]))

indist = f1("test"); rt = f1("rt"); am = f1("am")
mean_ood = (rt + am) / 2
print(indist, rt, am, mean_ood)
