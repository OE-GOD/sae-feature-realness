"""Coverage-matched head-to-head: graded ENSEMBLE-of-reliable-submodels vs the BINARY
agreement champion. Removes the coverage artifact (both judged at the champion's natural
coverage) and tests the AUGRC margin with a paired instance bootstrap."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from idea_harness import H, OOD, _auroc, _augrc

rng = np.random.RandomState(0)
B_SUB = 15      # bootstrap submodels
B_BOOT = 2000   # bootstrap resamples for significance

print(f"{'pool':6}{'dom':7}{'cov':>6} | {'champ':>7}{'ens@cov':>8}{'Δacc':>7} | "
      f"{'champAUC':>9}{'ensAUC':>8} | {'champGRC':>9}{'ensGRC':>8}{'Δgrc':>7}{'P(ens<)':>9}")
rows = []
for pool in ["mean", "max"]:
    h = H(pool)
    rel = h.reliable
    relp = h.probe(cols=rel)                       # single reliable probe (the champion's)
    subs = []                                      # 15 bootstrap reliable probes (fit once per pool)
    for b in range(B_SUB):
        idx = rng.randint(0, len(h.ytr), len(h.ytr))
        subs.append(LogisticRegression(max_iter=500, C=0.3).fit(h.Ztr[idx][:, rel], h.ytr[idx]))
    for dom, (P, yt) in OOD.items():
        Z = h.std(P[pool].float().numpy()); y = yt.numpy()
        fp = h.full.predict(Z); correct = (fp == y).astype(int); n = len(y)
        # champion: binary agreement of single reliable probe with full
        champ_bin = (fp == relp.predict(Z[:, rel])).astype(float)
        keep_c = champ_bin > 0.5; cov = keep_c.mean()
        champ_acc = correct[keep_c].mean()
        # ensemble: fraction of 15 submodels agreeing with full (continuous trust)
        agree = np.zeros(n)
        for s in subs:
            agree += (s.predict(Z[:, rel]) == fp)
        ens = agree / B_SUB
        # matched coverage: keep top-k by ens (jitter for fair tie-break), k = champion's kept count
        k = int(round(cov * n))
        jitter = rng.uniform(0, 1e-6, n)
        order = np.argsort(-(ens + jitter)); keep_e = np.zeros(n, bool); keep_e[order[:k]] = True
        ens_acc = correct[keep_e].mean()
        # coverage-independent metrics
        cA, eA = _auroc(champ_bin, correct), _auroc(ens, correct)
        cG, eG = _augrc(champ_bin, correct), _augrc(ens, correct)
        # paired instance bootstrap: P(ensemble AUGRC < champion AUGRC)
        wins = 0
        for _ in range(B_BOOT):
            bi = rng.randint(0, n, n)
            if _augrc(ens[bi], correct[bi]) < _augrc(champ_bin[bi], correct[bi]): wins += 1
        p = wins / B_BOOT
        rows.append((cA, eA, cG, eG, champ_acc, ens_acc, p))
        print(f"{pool:6}{dom:7}{cov:>6.2f} | {champ_acc:>7.3f}{ens_acc:>8.3f}{ens_acc-champ_acc:>+7.3f} | "
              f"{cA:>9.3f}{eA:>8.3f} | {cG:>9.4f}{eG:>8.4f}{cG-eG:>+7.4f}{p:>9.2f}")

r = np.array(rows)
print("\n--- SUMMARY (6 conditions) ---")
print(f"matched-coverage selective acc: ensemble beats champion in {int((r[:,5]>r[:,4]).sum())}/6 "
      f"(mean Δacc {np.mean(r[:,5]-r[:,4]):+.3f})")
print(f"AUROC: ensemble higher in {int((r[:,1]>r[:,0]).sum())}/6 (mean {np.mean(r[:,0]):.3f} -> {np.mean(r[:,1]):.3f})")
print(f"AUGRC: ensemble lower(better) in {int((r[:,3]<r[:,2]).sum())}/6 (mean margin {np.mean(r[:,2]-r[:,3]):+.4f})")
print(f"paired bootstrap P(ensemble AUGRC < champion): per-condition {np.round(r[:,6],2)}; "
      f"min {r[:,6].min():.2f}, mean {r[:,6].mean():.2f}")
