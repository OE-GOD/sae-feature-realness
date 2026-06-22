# Does interpretability know when a model is wrong?
### The reliable-feature account of out-of-distribution trust, and the resourced study to test it at scale

*A standalone research lead, grounded in a completed laptop-scale study with proven, verified,
and honestly-bounded results. This version supersedes the earlier draft: the central question is
no longer open — it is answered for one regime, and the answer comes with a mechanism, a theorem,
and a precise boundary. The resourced study tests whether that answer holds where it matters.*

---

## 1. What the laptop study established (verified, with theory)

**The principle.** For an out-of-distribution prediction, trustworthiness is carried by **whether
transfer-stable features support the call.** A prediction built on features that don't survive the
shift is the untrustworthy one — measurable with no labels.

**Proven.** A detector's error-*ranking* (AUROC) is invariant to any global monotone rescaling of
its score, so temperature/Platt calibration cannot repair plain confidence as an error signal. And
confidence empirically *collapses* under shift: across 6 OOD domains, corr(detector accuracy,
confidence's error-AUROC) = **+0.87** — it only knows it is wrong when the model is already right.

**Mechanism (verified, and corrected from my first guess).** Restricting the probe to its
transfer-stable features yields a **strictly better OOD oracle** (mean accuracy **0.72 → 0.82**;
random restriction 0.62). The trust signal works by comparing the deployed model to this better
oracle: where the call rests on shift-fragile features and the reliable view objects, it is likely
wrong. Directly confirmed — on near-chance financial data, *wrong* predictions have reliable
features pushing **against** them; *correct* ones have reliable features backing them. The original
"independent failure modes" story is empirically false; "better oracle" is the right account.

**Best operationalization.** A single-probe **reliable-evidence attribution** — score a prediction
by how much of its own margin comes from reliable features — beats confidence (domain-clustered
bootstrap AUROC gap **+0.234, P = 1.00**) and edges refit-based variants on harder shifts
(**+0.05, P = 0.98**). Disagreement, surprisal, and evidence-attribution correlate ~0.6: three
forms of one principle, the simplest best.

**The safety-relevant property: graceful degradation.** The interpretability signal's advantage
over confidence **doubles as the model gets worse** (gap +0.13 when accuracy ≥ 0.75, +0.28 when
< 0.65). On a near-chance shift you cannot fix the call, but you can still reliably gate it.

**The boundary (a theorem-backed correction).** This is **not** a universal law. Marginal
density/novelty *is* a valid error signal in an **extrapolation regime** (accuracy degrades with
distance from training support) — verified counterexample, AUROC 0.81, no class contrast. It fails
on the register/vocabulary shifts studied (AUROC 0.48–0.60) only because those are
**boundary-dominated**, so novelty encodes register-distance, not wrongness. A label-free
diagnostic distinguishes the regimes: regress novelty on margin over the unlabeled test batch.

**Honest limits.** (i) These are *ranking* results; label-free keep/abstain *thresholds* break
under shift (split-conformal coverage drifts to 0.42–0.58 vs nominal 0.50). (ii) Scope: one model,
one layer, sentiment, 6 domains, small n.

## 2. The question the resourced study answers

**Does the reliable-feature account of OOD trust hold where it matters — on instruct models, for
safety concepts, during generation — and can its known failure (label-free operating points) and
its regime boundary be resolved at scale?**

- **H1 (scale & safety):** reliable-evidence beats confidence at flagging wrong predictions for
  safety concepts (toxicity, jailbreak, PII, hallucination) on instruct models, with the same
  graceful-degradation property.
- **H2 (generation):** the principle extends from classification to **generation** — does an
  instruct model's next-token call, when carried by shift-fragile features, predict hallucination?
  (Adjacent to Ferrando et al., "Do I Know This Entity?", ICLR 2025.)
- **H3 (regime boundary at scale):** on real *extrapolation*-type shifts, density becomes the error
  signal and reliable-evidence weakens — and the label-free diagnostic predicts which regime holds
  before deployment.
- **H4 (operating points, the open problem):** can any method deliver label-free coverage control
  under shift, where standard conformal fails?

## 3. Method

Head-to-head at scale: plain confidence vs reliable-evidence (and the disagreement/surprisal
variants) on the error-detection task, across {≥2 instruct models} × {≥4 safety concepts} ×
{≥4 OOD domains per concept, spanning both boundary-dominated and extrapolation shifts}. Extend to
generation via token-level reliable-evidence and a hallucination oracle. Primary metric: AUROC and
**AUGRC** (Traub/Jäger, NeurIPS 2024) for error-ranking; selective accuracy at matched coverage;
graceful-degradation slope (advantage vs base accuracy). Real classifiers/oracles only; held-out
and cross-distribution; paired and **domain-clustered** significance; leakage audits; the
random-feature control throughout. Pre-registered, with the laptop study's discipline (including
public self-correction).

## 4. Outcomes (all publishable)

- **Holds:** interpretability provides a deployable, mechanism-backed OOD trust monitor whose edge
  grows where the model is weakest — the positive safety contribution.
- **Regime-bounded:** the boundary-dominated vs extrapolation split governs which signal to use,
  with a label-free diagnostic — a usable map, not a single trick.
- **Fails at scale:** a clean negative that redirects the field (e.g., the account is sentiment- or
  small-model-specific) — still worth reporting.

## 5. Why I'm positioned to run it

The laptop study already (a) established the principle with a proven theorem and a verified
mechanism, (b) found and verified the best operationalization, (c) mapped the regime boundary with a
reproduced counterexample, (d) built the full detector + cross-distribution + harder-shift pipeline,
and (e) practiced adversarial self-checking — four fleets built to break my own claims, two of
which succeeded and forced corrections now public. This is the precise, resourced next step, not a
fresh start.

## 6. Resources

GPU for instruct-model probing, generation, and abstention eval at scale; released SAE suites
(Gemma Scope, Llama Scope); labeled safety datasets with multiple OOD domains of *both* shift types;
judge models / real classifiers as oracles. Primarily inference + analysis — tractable within a
focused fellowship-scale project.

## 7. Positioning

Against Kantamneni et al. ("Are SAEs Useful?", ICML 2025) — SAE structure can be load-bearing for
**selective risk** even where it is not for accuracy; against Agreement-on-the-Line (Baek et al.,
NeurIPS 2022) — per-instance, model-vs-its-reliable-self, not two independent full models; alongside
Ferrando et al. (ICLR 2025) for the generation extension; using AUGRC (Traub/Jäger, NeurIPS 2024)
and reporting where conformal validity breaks under shift.
