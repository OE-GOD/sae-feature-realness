# Does interpretability know when a model is wrong?
### Testing whether interpretability-native signals beat plain confidence for out-of-distribution abstention

*A standalone research lead. One falsifiable question, grounded in a completed laptop-scale study.*

---

## The question

When a model's prediction can't be trusted out-of-distribution, **can an interpretability signal
flag it better than the model's own confidence can?** Or is plain confidence all you need, and
interpretability adds nothing to knowing-when-wrong?

## Why it matters, and why now

A detector you deploy is, by definition, run on data unlike its training set. My laptop-scale study
([4 posts + repo](https://github.com/OE-GOD/sae-feature-realness)) established two things:

1. **You can't reliably make SAE-feature detectors *accurate* out-of-distribution** — 11 methods, no consistent win.
2. **But you can make them trustworthy by abstention** — abstaining on low-confidence OOD inputs raised
   accuracy 0.73→0.83 across three domains (verified vs random abstention).

The catch — and the opening for this project: **the working abstention signal was plain classifier
confidence. The interpretability-native signal I tried (abstain when the input activates training-rare
features) failed at chance (AUROC 0.51).** So the field's hope — that interpretability gives a privileged
view of *when a model is operating outside its competence* — is, so far, unsupported.

That sets up a clean, high-stakes question with a binary answer: **either a properly-designed
interpretability signal beats plain confidence at knowing-when-wrong (interpretability earns its keep for
safety monitoring), or it doesn't (a sharp, useful negative that redirects the field to selective
prediction).** The naive version is already ruled out, so the next test can be aimed precisely.

## Hypothesis and the bar

**H:** an interpretability-native abstention signal can match or beat plain softmax confidence at
separating a model's correct OOD predictions from its wrong ones.

**The bar is explicit and brutal:** plain confidence is the champion (it won at toy scale). An
interpretability signal counts only if it *beats* confidence on held-out OOD — not if it merely "helps."

## The three signals to test (each fixes a specific toy failure)

1. **Attribution-weighted confidence.** *Toy failure:* the novelty signal weighted all active features
   equally (noise). *Fix:* weight by each feature's reliability (cross-seed replication) and its causal
   contribution to *this* prediction. Intuition: trust a prediction only if *reliable* features drove it.
2. **Circuit-consistency.** *New, purely interpretable signal:* does the prediction route through the same
   circuit it uses in-distribution? OOD inputs that reach the answer via the *wrong path* are
   untrustworthy even when the output looks confident. No analog exists in plain confidence.
3. **Reliable-subspace novelty.** *Toy failure:* raw distance over 16k dims was at chance. *Fix:* measure
   novelty only in the low-rank subspace of the *reliable* features, with a learned threshold.

## Method

Head-to-head at scale: plain confidence vs the three interpretability signals, on the abstention task,
across {≥2 instruct models} × {≥4 safety-relevant concepts, e.g. toxicity / jailbreak / PII / refusal} ×
{≥4 OOD domains per concept}. Primary metric: **selective-accuracy AUROC** (does the signal rank correct
predictions above wrong ones OOD), plus accuracy-vs-coverage curves and silent-failure rate. Pre-registered
discipline carried from the prior study: real classifiers/oracles only, held-out + cross-distribution,
paired significance, leakage audits, and **always beat the simple baseline (plain confidence).**

## Outcomes (both publishable)

- **Interpretability wins:** a signal beats confidence → interpretability provides genuine, deployable
  value for safety monitoring (knowing when a model is outside its competence). The positive contribution.
- **Interpretability loses:** none beat confidence across the board → a clean negative — *for OOD trust,
  use selective prediction, not SAE-feature signals* — which saves the field real effort and is itself a
  finding worth reporting.

## Why I'm positioned to run it

The laptop study already (a) ruled out the naive interpretability signal, (b) verified the abstention
result it must beat, (c) built the detector + cross-distribution + abstention pipeline, and (d) practiced
the verification discipline (including publicly retracting a result that failed re-testing). This project
is the precise, resourced next step — not a fresh start.

## Resources / feasibility

GPU for instruct-model probing + abstention eval at scale; released SAE suites (Gemma Scope, Llama Scope);
labeled safety datasets (toxicity, jailbreak, PII) with multiple OOD domains; judge models / real
classifiers as oracles. No model training required beyond light probing — primarily inference + analysis,
so it is tractable within a focused fellowship-scale project.
