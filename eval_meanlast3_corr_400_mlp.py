import torch, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "meanlast3"; nf = 400

def getX(split): return data[split][0][pooling].float().numpy()
def gety(split): return data[split][1].numpy()

Xtr = getX("train"); ytr = gety("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
def std(X): return (X - mu) / sd
Xtr_s = std(Xtr)

# corr selection: |point-biserial corr| with ytr
yc = (ytr - ytr.mean())
ystd = yc.std()
corr = np.abs((Xtr_s * yc[:, None]).mean(0) / ystd)  # Xtr_s already unit-std per col
cols = np.argsort(-corr)[:nf]

Xtr_sel = Xtr_s[:, cols]
clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
clf.fit(Xtr_sel, ytr)

def ev(split):
    Xs = std(getX(split))[:, cols]
    return f1_score(gety(split), clf.predict(Xs))

indist = ev("test"); rt = ev("rt"); am = ev("am")
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
