import torch, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "meanlast3", 400

def X(split): return data[split][0][pooling].float().numpy()
def y(split): return data[split][1].numpy()

Xtr, ytr = X("train"), y("train")
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)

mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def evalf1(split):
    Xs = scaler.transform(X(split))[:, cols]
    return f1_score(y(split), probe.predict(Xs))

indist = evalf1("test")
rt = evalf1("rt")
am = evalf1("am")
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
