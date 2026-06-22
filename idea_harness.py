"""Fixed, validated harness for testing new OOD-abstention signals on EXISTING
precollected SAE-feature data (CPU only). Every candidate idea plugs in the same
way, so results are comparable and the leak-free protocol can't drift per-agent.

Contract for an idea:
    def make_signal(H, pool):
        # build whatever you need from H (train/rt only — NEVER test OOD domains)
        # return score(Ztest) -> np.array of TRUST scores (higher = keep/trust)
        return score
    H.run(make_signal, "my idea name")

H.run reports, for every (pooling x OOD domain): AUROC(trust vs correct),
AUGRC (lower=better), and selective accuracy at the candidate's natural coverage
with PLAIN CONFIDENCE and the RANDOM-FEATURE control held to the same coverage.
A real win = beats BOTH confidence and random on selective accuracy, leak-free.
"""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression

_d = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
_more = torch.load("/Users/oe/rebuild/more_ood_sentiment.pt", weights_only=False)
POOLINGS = ["mean", "max"]  # the poolings present across all OOD files (yelp/imdb lack last/meanlast3)
OOD = {"amazon": _d["am"], "yelp": _more["yelp"], "imdb": _more["imdb"]}

def _scorr(Z, y):
    yc = y - y.mean(); zc = Z - Z.mean(0)
    return np.nan_to_num((zc * yc[:, None]).mean(0) / (zc.std(0) * yc.std() + 1e-9))

def _augrc(trust, correct):
    # area under generalized risk-coverage curve (Traub/Jaeger 2024): avg risk of
    # undetected failures; lower is better. order by descending trust, breaking ties
    # RANDOMLY (deterministic seed) — row-order tie-breaks biased binary signals when
    # the data is label-ordered (audit, June 2026).
    tie = np.random.RandomState(0).permutation(len(trust))
    order = np.lexsort((tie, -trust)); c = correct[order].astype(float)
    n = len(c); cov = np.arange(1, n + 1) / n
    risk = np.cumsum(1 - c) / n            # generalized risk: errors / TOTAL, not /kept
    return float(np.mean(risk))            # area under (coverage, generalized-risk)

def _auroc(trust, correct):
    # P(trust(correct) > trust(wrong))
    pos = trust[correct == 1]; neg = trust[correct == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

class H:
    """Per-pooling context handed to make_signal. Train/rt only for building signals."""
    def __init__(self, pool):
        self.pool = pool
        Xtr = _d["train"][0][pool].float().numpy()
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        self.ytr = _d["train"][1].numpy()
        self.yrt = _d["rt"][1].numpy()
        self.Ztr = (Xtr - self.mu) / self.sd
        self.Zrt = (_d["rt"][0][pool].float().numpy() - self.mu) / self.sd
        freq = (Xtr > 0).mean(0)
        self.alive = np.where(freq > 0.01)[0]
        c_tr = _scorr(self.Ztr, self.ytr); c_rt = _scorr(self.Zrt, self.yrt)
        self.feat_stability = c_tr * c_rt           # high = sign-stable across 2 in-dist domains
        self.reliable = self.alive[np.argsort(-(c_tr[self.alive] * c_rt[self.alive]))[:500]]
        self.full = LogisticRegression(max_iter=500, C=0.3).fit(self.Ztr, self.ytr)
    def std(self, Xraw):  # standardize a raw OOD pooling matrix with TRAIN stats
        return (Xraw - self.mu) / self.sd
    def probe(self, cols=None, C=0.3):
        Z = self.Ztr if cols is None else self.Ztr[:, cols]
        return LogisticRegression(max_iter=500, C=C).fit(Z, self.ytr)
    def scorr(self, Z, y): return _scorr(Z, y)

def run(make_signal, name):
    print(f"\n=== {name} ===")
    print(f"{'pool':9}{'domain':8}{'base':>6}{'AUROC':>7}{'AUGRC':>7}"
          f"{'cov':>6}{'CAND':>7}{'conf':>7}{'rand':>7}{'win?':>6}")
    wins = 0; total = 0; rows = []
    for pool in POOLINGS:
        h = H(pool)
        try:
            score = make_signal(h, pool)
        except Exception as e:
            print(f"{pool:9}make_signal FAILED: {e}"); continue
        for dom, (P, yt) in OOD.items():
            Z = h.std(P[pool].float().numpy()); y = yt.numpy()
            pf = h.full.predict(Z); correct = (pf == y).astype(int); base = correct.mean()
            trust = np.asarray(score(Z), dtype=float)
            au = _auroc(trust, correct); ag = _augrc(trust, correct)
            # natural coverage: if binary trust, coverage = frac kept; else top-50%
            uniq = np.unique(trust)
            if len(uniq) <= 2:
                keep = trust >= uniq.max(); cov = keep.mean()
            else:
                cov = 0.5; thr = np.quantile(trust, 1 - cov); keep = trust >= thr
            cand_acc = correct[keep].mean() if keep.sum() else base
            conf = np.abs(h.full.decision_function(Z))
            cthr = np.quantile(conf, 1 - cov); conf_acc = correct[conf >= cthr].mean() if (conf>=cthr).sum() else base
            rnds = []
            for s in range(5):
                rr = np.random.RandomState(s).choice(h.alive, len(h.reliable), replace=False)
                rp = h.probe(cols=rr); ra = (pf == rp.predict(Z[:, rr]))
                rnds.append(correct[ra].mean() if ra.sum() else base)
            rand_acc = np.mean(rnds)
            win = cand_acc > conf_acc + 1e-9 and cand_acc > rand_acc + 1e-9
            wins += win; total += 1
            rows.append((pool, dom, au, ag, cov, cand_acc, conf_acc, rand_acc, win))
            print(f"{pool:9}{dom:8}{base:>6.3f}{au:>7.3f}{ag:>7.3f}{cov:>6.2f}"
                  f"{cand_acc:>7.3f}{conf_acc:>7.3f}{rand_acc:>7.3f}{'  Y' if win else '  .':>6}")
    print(f"--- {name}: beats BOTH confidence & random on {wins}/{total} conditions ---")
    return {"name": name, "wins": wins, "total": total, "rows": rows}

H.run = staticmethod(run)  # convenience: H.run(make_signal, name)
