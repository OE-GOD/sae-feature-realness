---
title: "Are SAE Features Real? From an Unanswerable Question to a Deployable Detector"
author: Aung "OE" Maw
date: 2026-06-17
---

## TL;DR

I tried to answer "are sparse-autoencoder features real?" and found the question, as
usually asked, is unanswerable — there is no oracle for which features are real, and
when you test it, most individual features fail. But the *useful* version is answerable:
you can build, validate, and deploy feature-based **detectors**, certified by held-out
and cross-distribution task performance. I built the full pipeline from scratch (~22
scripts), reproduced my own published replication finding four times, hit the no-oracle
wall empirically, and ended with a deployed detector that scores perfect precision on
unseen prose and **fails safe** on code. Bottom line: **don't trust individual features;
build validated combinations, and document their scope.**

---

## Why I rebuilt everything from scratch

I had a portfolio of interpretability work that AI had largely typed for me. I could
*recognize* it but not *reproduce* it — and that gap showed the moment I was asked to
explain it under pressure. So I rebuilt the core by hand: trained SAEs on a language
model, typed every line, justified every step before running it. This writeup is what
that rebuild found when I pointed it at one question: **are the features real?**

All experiments are reproducible scripts in `/Users/oe/rebuild/`.

---

## Finding 1 — Most individual features don't replicate

Train two SAEs on the same model activations, identical except the random seed. If a
feature is a real property of the model, both runs should find it.

- Toy SAE (Pythia-160M L6, 2048 features): both SAEs reached **identical loss (0.1056)**,
  yet only **1.0%** of features had a cross-seed twin at cosine > 0.9.
- My earlier 16k study found **2.14%** across seed/width/architecture.

Equally good at the job; almost no shared vocabulary. **Most "features" are artifacts of
one training run.** (`03_sae.py`, `05_replicate.py`)

But — this does *not* mean the model has no real concepts. The information is there; it's
**smeared across many features and sliced differently each seed**. The concept is real;
the single-feature label for it usually isn't.

---

## Finding 2 — "Real" is not one property. It's several, and they disagree.

I operationalized five independent tests of "real":

| Axis | Question | How |
|---|---|---|
| Stability | does it replicate across seeds? | cross-seed decoder cosine |
| Necessity | does removing it change output? | ablation (per-firing) |
| Sufficiency | does adding it induce the concept? | steering |
| Interpretability | does it fire on one consistent thing? | top-token concentration |
| Logit-coherence | clean direct effect on output tokens? | decoder · unembedding |

Then I asked whether the axes agree. They mostly don't.
(`16`, `17`, `19`, `20`, `21`)

- **stability ↔ causation:** weakly negative (~ -0.2 at high N; stronger on Gemma).
  The features that replicate are often the ones the model uses *least* — my old
  atom/manifold split, reproduced.
- **interpretability ↔ stability:** ~0 (independent). A feature can fire 100% on
  newlines (perfectly interpretable) yet be totally unstable (cos 0.17).
- **logit-coherence:** stands alone, independent of interpretability and causation.
- **necessity ≠ sufficiency:** ablating and steering measure different things.
- **decoder-norm geometry:** stability +0.49, frequency -0.48 — big-norm directions are
  stable but causally inert; small-norm are the causal ones.

A feature can pass one test and fail another. **One test is never enough.**

A noise warning I had to heed: at n=40, individual correlations wobbled ±0.3
(interp↔causation went +0.37 → +0.06 between runs). I re-ran the cheap axes at n=723 to
get trustworthy numbers, and walked back my own earlier "strong" claims to "weak."

---

## Finding 3 — Cheap signals predict *stability* but not *causation*

Firing frequency (free, one SAE) predicts stability moderately (Pearson **0.52** on
Pythia, **0.54** at 5 seeds, **0.37** on Gemma) — via an inverted-U: rare features are
hyper-specific fragments, common ones are incoherent background, clean concepts sit in
between. (`08`, `09`, `15`)

But frequency does **not** predict causation (confound-controlled per-firing: ~ -0.1).
**You cannot cheaply tell whether the model uses a feature.** The causal axes are
independent of every cheap signal — they must be measured directly. (`11`)

---

## Finding 4 — The no-oracle wall (why individual-feature certification is impossible)

To certify a test, you validate it against ground truth. For "real features," there is
no ground truth — that's the question itself.

- On **synthetic** data where I planted known features, the battery worked: stability
  test **precision 1.00**, recall 0.62; necessity precision 0.70, recall 0.82. (`synthetic_validate.py`)
- On **real** activations with injected ground truth, precision **collapsed to 0.04** —
  not because the test failed, but because real activations contain Pythia's own
  unlabeled real features, which the test correctly flags but my oracle can't label.
  And only **6/15** injected-real directions were even recovered amid superposition.
  (`inject_validate.py`)

This is the wall, demonstrated: **you can't measure certifier precision on real features
because validating it requires the complete list of real features you're trying to find.**
A full 4-axis battery certified **0/12** toy features. (`certify.py`)

---

## The pivot — from feature-realness to detector-validation

If you can't certify individual features, change the question. Don't ask "is this feature
real?" Ask "does this **validated combination** of features reliably do a task?" A task
has labels — the labels are the oracle. This sidesteps the wall entirely.

**Method (the recipe):** define a labeled task → get SAE activations → split held-out →
select top-k correlated features → fit a sparse probe → tune threshold on train → judge
on held-out → certify only if it clears the bar. (`build_detector.py`)

- Single features failed (digit recall 0.14 — splitting). (`gemma_downstream.py`)
- Sparse probes (top-30 features + logistic regression) on the **real, deployable**
  Gemma Scope SAE certified **3/5** detectors at held-out F1 > 0.8:
  digit 0.83, newline 0.88, space 0.85. (`gemma_probe.py`)

---

## Industrial validation — detectors must hold across distributions

Held-out same-distribution isn't enough for deployment. I tested Pile-built detectors on
TinyStories (a different distribution). (`industrial_validate.py`)

| Detector | in-dist F1 | OOD F1 | Verdict |
|---|---|---|---|
| newline | 0.88 | 0.98 | CERTIFIED (holds cross-distribution) |
| space | 0.85 | 0.98 | CERTIFIED |
| digit | 0.84 | 0.00 | **REJECTED** — no digits in OOD; would break in deployment |

The cross-distribution test **caught a detector that passed in-distribution but would
have silently failed in production.** Catching that is the entire value of validation.

---

## Real-use deployment — works in scope, fails safe out of scope

I loaded the saved `detector_newline.json` and ran it on fresh input. (`deploy_demo.py`)

- Fresh unseen Pile prose: **precision 1.00, recall 1.00.**
- Python code (untested domain): **precision 1.00, recall 0.40** — degrades, but
  **fails safe**: no false positives, just misses code newlines (structurally different).

Refined insight: distribution shift hurts in proportion to **token-structure** difference,
not topic. prose→prose (TinyStories) transfers (0.98); prose→code does not (0.57).

---

## Conclusion

> **Individual SAE features mostly aren't real — but the model's concepts are. They live
> in combinations of features, not single units. So don't certify features; build, validate,
> and deploy feature-based detectors — judged by held-out and cross-distribution task
> performance, documented with their scope and failure modes.**

This turns an unanswerable ontological question ("are features real?") into an answerable
engineering one ("does this validated detector work, and where?"). The shift from
**feature-realness to detector-validation** is the contribution.

## Honest limitations

- Toy/simple concepts (newline, digit, space). High-stakes concepts (toxicity, PII) are
  harder and need their own labeled tasks — but the *method* is identical.
- One SAE, one layer; cross-distribution tested on one OOD corpus.
- Certifies a *detector's* reliability, NOT that its constituent features are individually
  "real" — a deliberately weaker, honest claim.
- The no-oracle wall is fundamental, not an engineering gap: native-feature realness
  cannot be certified with precision guarantees on real models.

## Reproducibility

~22 scripts in `/Users/oe/rebuild/` (`01_look` → `deploy_demo`), plus `VALIDATION_REPORT.md`
(the detector certificate) and `detector_newline.json` (a deployable artifact). Built by
hand; every number above is from these runs.
