# Detector Validation Report — SAE Feature Detectors for Industrial Use
SAE: gemma-scope-2b-pt-res / layer_12 / width_16k / average_l0_82 (deployed artifact)
Method: sparse probe (top-30 features + logistic regression), validated by held-out
        AND cross-distribution F1. Oracle = task labels. Build corpus: The Pile.

## Validation protocol (what "industrial-certified" means here)
1. Build detector on Pile docs 0-59.
2. IN-DISTRIBUTION test: Pile docs 60-89 (unseen docs, same distribution).
3. OUT-OF-DISTRIBUTION test: TinyStories (a genuinely different distribution).
4. CERTIFY only if in-dist F1 > 0.80 AND OOD F1 > 0.70.
   (A detector that passes in-dist but fails OOD is REJECTED — it would break in deployment.)

## Results
| Detector  | in-dist F1 | OOD F1 | Status |
|-----------|-----------|--------|--------|
| newline   | 0.88 | 0.98 | CERTIFIED — robust across distributions |
| space_pre | 0.85 | 0.98 | CERTIFIED — robust across distributions |
| cap_word  | 0.80 | 0.85 | borderline (in-dist at bar) |
| digit     | 0.84 | 0.00 | REJECTED — distribution-dependent (no digits in OOD) |
| punct     | 0.70 | 0.83 | REJECTED — unreliable in-distribution |

## Certified detectors (deployable)
- **newline**: 30 features, held-out precision 0.96 / recall 0.83, robust on TinyStories.
  Artifact: detector_newline.json
- **space_pre**: 30 features, in-dist F1 0.85, OOD F1 0.98.

## Documented failure modes (the honest part — required for industrial trust)
- **digit detector is distribution-dependent**: certified on Pile, scores 0.00 on a
  no-digit distribution. DO NOT deploy without validating on the target distribution.
- All certifications are SCOPED to (a) this exact SAE, (b) these concepts, (c) tested
  distributions. A new domain (code, other languages) requires re-validation.
- Detectors are feature-COMBINATIONS, not single features. They certify the detector's
  reliability, NOT that any individual feature is "real."

## Bottom line
Individual SAE features cannot be certified real. Validated feature-COMBINATIONS can:
certify a detector by held-out + cross-distribution task performance, document its scope
and failure modes, and re-validate per target distribution. This report is the template.

## Real-use deployment test (newline detector, loaded from detector_newline.json)
- Fresh unseen Pile prose (doc 500): precision 1.00, recall 1.00 — PERFECT in-scope.
- Python code (untested domain): precision 1.00, recall 0.40 — degrades but FAILS SAFE
  (no false positives; misses structurally-different code newlines).
- Insight: distribution shift hurts in proportion to TOKEN-STRUCTURE difference, not topic.
  prose->prose (TinyStories) transfers (F1 0.98); prose->code does not (F1 0.57).
- Deployment guidance: deploy in-scope (prose); re-validate before using on code/new domains.
