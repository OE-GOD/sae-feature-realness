import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

POOL="last"; NF=100
data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

def X(split): return data[split][0][POOL].float().numpy()
def Y(split): return data[split][1].numpy()

Xtr, ytr = X("train"), Y("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1.0
Xtr_s = (Xtr-mu)/sd

mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:NF]

Xtr_sel = Xtr_s[:, cols]
clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
clf.fit(Xtr_sel, ytr)

def evalf(split):
    Xs = (X(split)-mu)/sd
    return f1_score(Y(split), clf.predict(Xs[:, cols]))

indist = evalf("test"); rt = evalf("rt"); am = evalf("am")
mean_ood = (rt+am)/2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
