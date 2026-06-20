# Are SAE features real? From an unanswerable question to validated detectors

Code for the investigation written up at
[oe-god.github.io/2026/06/17/are-sae-features-real](https://oe-god.github.io/2026/06/17/are-sae-features-real/).

Built from scratch by hand on Pythia-160M and Gemma-2-2b (Gemma Scope SAEs).
Every result in the writeups is produced by a script here.

## The one-line finding

Individual SAE features mostly can't be certified "real" (no oracle; ~1–2% replicate
across seeds). The trustworthy unit is a **validated detector** — a sparse combination of
features judged by held-out **and cross-distribution** task performance. Surface concepts
certify cleanly (F1 ~0.98 cross-distribution on a real SAE); semantic concepts improve to
~0.82 but are not cleanly cross-distribution-robust, and pooling is *not* the lever.

## Reproducing

```bash
# uses transformer_lens, sae_lens, datasets, torch, scikit-learn
python 01_look.py        # see a model's layer-6 activations
python 02_collect.py     # collect activations  -> thoughts.pt  (gitignored, regenerate)
python 03_sae.py         # train a TinySAE       -> sae.pt
python 05_replicate.py   # train seed-2 SAE, cross-seed replication rate (~1%)
```
Large data/checkpoints (`*.pt`) are gitignored — regenerate them with the
`*precollect*.py` / `03_sae.py` scripts.

## Map of the code

**The rebuild (learn the stack by hand), 01–21:**
- `01_look` activations · `02_collect` data · `03_sae` train · `04_inspect` features
- `05_replicate` cross-seed replication · `06_ablate` causal test
- `07–11` realness axes (stability/causation/interpretability/frequency, confound-controlled)
- `12–18` Gemma cross-architecture + the stability⊥causation structure
- `19–21` logit-coherence axis + full axis-correlation matrix

**Realness tooling:**
- `certify.py`, `feature_realness.py`, `feature_realness_ranked.py` — multi-axis feature
  scoring + ranked triage with causal spot-check
- `synthetic_validate.py`, `inject_validate.py` — the no-oracle wall, demonstrated

**Detector validation (the contribution):**
- `build_detector.py` — build + validate + save a detector for any concept
- `gemma_best_detectors.py` — winning recipe (L1-select + MLP) on real Gemma Scope SAE
- `industrial_validate.py`, `cross_dist_scale.py` — cross-distribution validation
- `deploy_demo.py` — load a saved detector, run on fresh text
- `safety_detector.py`, `pii_detector.py`, `frontier_hate.py` — semantic/safety frontier
- `detect_tool_and_semantic.py` — usable tool + semantic sentiment test

**Agent-fleet experiment records:** `eval_*.py`, `run_*.py` are scratch scripts from
parallel recipe searches (selection × #features × probe × pooling). The curated entry
points above supersede them; they're kept for the record.

**Writeups:** `WRITEUP.md` (full), `CORE.md` (9 core concepts), `RECAP.md`,
`VALIDATION_REPORT.md`. Figures: `fig*.png`.

## Honest scope

Toy SAE for the rebuild; real Gemma Scope SAE for the detector results. Simple/surface
concepts certify; semantic concepts are improved but not solved. This certifies a
*detector's* reliability, not that individual features are "real." See `WRITEUP.md` for
the calibrated claims and the SAEBench positioning.
