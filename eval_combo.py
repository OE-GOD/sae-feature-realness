import torch, numpy as np
from sklearn.linear_model import LogisticRegression
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

# l1 selection: top-nf by |coef| from L1 logistic regression
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr_s, ytr)
cols = np.argsort(-np.abs(sel.coef_[0]))[:nf]

# mlp probe
probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_s[:, cols], ytr)

def f1(split):
    return f1_score(Y(split), probe.predict(Z(X(split))[:, cols]))

indist = f1("test"); rt = f1("rt"); am = f1("am")
mean_ood = (rt + am) / 2
print(indist, rt, am, mean_ood)
