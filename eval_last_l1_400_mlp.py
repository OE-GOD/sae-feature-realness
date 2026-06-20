import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "last", 400

def X(split): return data[split][0][pooling].float().numpy()
def y(split): return data[split][1].numpy()

Xtr, ytr = X("train"), y("train")
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
def std(M): return (M - mu) / sd
Xtr_s = std(Xtr)

# l1 selection
l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
l1.fit(Xtr_s, ytr)
cols = np.argsort(np.abs(l1.coef_[0]))[::-1][:nf]

probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_s[:, cols], ytr)

def ev(split):
    return f1_score(y(split), probe.predict(std(X(split))[:, cols]))

indist = ev("test"); rt = ev("rt"); am = ev("am")
print("recipe last|l1|nf400|mlp")
print("indist", indist, "rt", rt, "am", am, "mean_ood", (rt+am)/2)
