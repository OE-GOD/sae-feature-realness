# Research proposal: Calibrated abstention for trustworthy SAE-feature signals

*Grounded in a completed laptop-scale study (4 posts, ~12 experiments). This is the
resourced study that goes further — what the toy work established, what it couldn't,
and what a GPU-scale program would resolve.*

## 1. The gap (what's established, what's open)

From the laptop study (Gemma-2-2b base, one layer, sentiment, 3 OOD domains):
- **Established (negatives):** most individual SAE features don't replicate (~2%); making a
  feature-detector *accurate* out-of-distribution is hard — 11 methods (de-spuriousing,
  generalization predictor, sign-stability, multi-domain training) gave small/inconsistent/no gains.
- **Established (positive):** trustworthiness is achievable as **calibrated abstention** — a detector
  that abstains on its least-confident OOD inputs raises accuracy ~0.73→0.83 across 3 domains.
- **Open / unresolved at toy scale:**
  1. Does the abstention result hold at scale, on **instruct models**, for **safety concepts**?
  2. The **interpretability-native** abstention signals (feature-novelty, Mahalanobis) failed at
     chance — can a better-designed interp signal beat plain confidence, or is confidence really all there is?
  3. Can OOD reliability be **predicted** in advance (toy predictor was weak, r≈0.49)?

The field independently names these: *"finding features ≠ confirming legitimacy; current methods lack
formal guarantees"* and *"need quantitative methods to predict feature generalization."*

## 2. Central question + hypotheses

**Q: Can interpretability give us detectors that are trustworthy out-of-distribution — not by being
accurate, but by reliably knowing when to abstain — and does interpretability add anything over plain
confidence?**

- **H1 (scale):** "accuracy OOD is hard, abstention OOD works" replicates across models
  (Gemma-2, Llama-3), layers, and concepts.
- **H2 (safety):** calibrated abstention yields a deployable safety detector (toxicity / jailbreak / PII)
  that abstains under distribution shift instead of failing silently — on an *instruct* model.
- **H3 (interp value):** a properly-designed interpretability-native abstention signal (feature-attribution
  to the prediction; activation-region novelty with learned weighting; circuit-level consistency) can
  match or beat plain softmax confidence OOD. (Toy version failed — H3 is the high-risk, high-reward arm.)
- **H4 (prediction):** with scale and richer signals, OOD reliability is predictable well above the toy r≈0.49.

## 3. Resources required (why this needs more than a laptop)

- GPU(s) for instruct-model generation + probing + (light) fine-tuning at scale.
- Models: Gemma-2-2b/9b-it, Llama-3-8b-Instruct; their released SAE suites (Gemma Scope, Llama Scope).
- Data: labeled safety datasets (toxicity: Jigsaw/RealToxicityPrompts; jailbreak: AdvBench/HarmBench;
  PII: synthetic + real); multiple OOD domains per concept; an adversarial/distribution-shift benchmark.
- Real classifiers / oracles for honest measurement (Detoxify, judge models), as in the toy work.

## 4. Experiments (arms)

**Arm A — Scale the core (H1).** Replicate "accuracy OOD hard / abstention OOD works" across
{2 models} × {3 layers} × {5 concepts} × {≥4 OOD domains each}. Establish the abstention effect with
proper selective-prediction curves (accuracy vs coverage), paired significance, multiple seeds.
Deliverable: is abstention-trustworthiness a general property of SAE-feature detectors?

**Arm B — Safety detectors that abstain (H2).** Build refusal/jailbreak/toxicity/PII detectors on an
instruct model; measure not just F1 but **selective F1 under distribution shift** and **silent-failure
rate** (confidently-wrong on OOD). The product question: can you ship a safety detector that says
"I don't know" off-distribution instead of failing silently? Compare to a fine-tuned classifier baseline.

**Arm C — Does interpretability add anything? (H3, the crux).** Head-to-head, at scale: plain confidence
vs interpretability-native abstention signals —
  (i) feature-attribution confidence (how much do *reliable* features drive this prediction?),
  (ii) learned activation-region novelty (not the unweighted toy version that failed),
  (iii) circuit-consistency (does the prediction route through the expected circuit?).
If none beats plain confidence, that's a clean, important negative for "interpretability for OOD trust."
If one does, it's the contribution.

**Arm D — Predicting reliability (H4).** Learn a predictor of a detector's OOD selective-accuracy from
in-distribution + interpretability features; validate on held-out concepts and models. Targets the
field's named gap directly.

## 5. Evaluation + falsification (pre-registered discipline)

- Real classifiers/oracles only; never proxy metrics (lesson from the retracted toxicity result).
- Every claim: held-out, cross-distribution, paired significance, vs the strongest simple baseline
  (the recurring finding: simple baselines are stubbornly strong — so they must be beaten, not ignored).
- Leakage audits (never select on the test domain) and degeneracy checks (no winning-by-incoherence).
- **Falsification up front:** H1 fails if abstention doesn't replicate; H3 fails if interp never beats
  plain confidence (a real, publishable negative either way).

## 6. Expected contributions

- A characterization (likely: "SAE-feature detectors can't be made accurate OOD but can be made to
  abstain reliably") at scale, across models and safety concepts.
- A deployable, honestly-validated **safety detector that abstains under shift** (Arm B) — the practical payoff.
- A clean answer to "does interpretability add anything to OOD trust beyond plain confidence?" (Arm C) —
  positive or negative, both valuable.
- A reliability predictor (Arm D), addressing a named open problem.

## 7. Risks / honest priors

- Most likely outcome: abstention generalizes (Arm A/B succeed), but interpretability does NOT beat plain
  confidence (Arm C negative) and prediction stays modest (Arm D weak). That's *still* a strong, honest
  paper: "trustworthy OOD interpretability = selective prediction; the interpretable signal isn't needed."
- The high-upside surprise: an interp-native signal (Arm C) or a circuit-consistency check genuinely beats
  confidence — which would be the real novel win.
- Either way the study resolves questions the laptop work could only pose.
