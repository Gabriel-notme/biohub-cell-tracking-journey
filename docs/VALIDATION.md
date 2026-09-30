# Validation protocol

A 0.001 change in this competition is often one division. Local numbers were also systematically optimistic, because the
public detector saw every training movie and all labelled movies come from two embryos. Most of my time therefore went into
deciding what to *believe*, not into building things. This page describes the protocol that emerged, and where it still
failed.

## 1. Ground rules

- **Official metric, official code.** I used `tracking_cellmot` from the organisers' repository at commit
  `075fc5f5a52d11077f9dc2b074644618f26939e2`. I never re-implemented the metric.
- **Score the artefact that would be submitted.** Every candidate is scored from the exact integer-coordinate
  `submission.csv` (or the graph it is exported from). Per-movie rows are aggregated the way the leaderboard aggregates:
  adjusted edge Jaccard weighted by TP+FP+FN, with divisions micro-averaged. Rounding coordinates matters: writing
  unrounded floats scored lower locally, and a forum report put the public cost at about 0.008.
- **Per-movie rows are kept.** `research/cl/review.py scoreg` writes one row per movie with edge TP/FP/FN, division TP/FP/FN
  and node counts. Every comparison is paired on these rows.

## 2. Movie groups

All 199 labelled movies were split once, and the split was never changed:

| Group | Movies | Status | Use |
|---|---:|---|---|
| `hold36` | 36 | the 18 calibration + 18 audit movies of b1 and B5 (B5 was calibrated on half of them); never used to train any P-stage model | clean held-out |
| `prev4` | 4 | the visible test preview movies | clean held-out; also used for Kaggle-vs-local checks |
| `audit32` | 32 | held out from B5's new heads; used to train P-stage linkers | semi-clean |
| `t127a` / `t127b` | 64 / 63 | B5 training movies; used to train P-stage linkers | in-sample |

- **clean40** = hold36 + prev4 is the cleanest estimate.
- **B5-unseen72** = hold36 + prev4 + audit32.
- The **embryo** split is 44b6 (71 movies) and 6bba (128 movies). Every comparison my tools produce reports both embryos
  separately.

The movie lists are in [`results/splits/`](../results/splits).

Known limitations:

- The public detector was trained on all 199 movies, so no group is truly clean for detection.
- hold36 is also an easy group: adjusted edge Jaccard is 0.945 there, against 0.92–0.93 for audit32 and t127.
- prev4 is dominated by one dense movie (`6bba_05db0fb1`, about 57 % of the edge weight).

## 3. Statistics

- **Paired movie bootstrap.** 300–2,000 resamples of movies, depending on the tool, give a 95 % CI and P(Δ > 0) for every
  delta.
- **Drop-top-k.** Recompute the delta after removing the k = 1, 3, 5 movies that gained most. This catches gains carried by
  one or two movies.
- **Better/worse movie counts,** both for all movies and for movies where edges or forks actually changed.
- **Division bookkeeping.** For each added or removed fork: is it on an annotated lineage ("evaluable"), and is it a TP or FP
  under the official division matcher (`research/cl/p21/div_diff.py`)?

### Noise model

From resampling labelled movies at leaderboard scale:

| Leaderboard | SD of one model's absolute score | SD of the difference between two related models |
|---|---:|---:|
| Public (~58 movies) | ≈ 0.017 | 0.0004–0.0011 |
| Private (~141 movies) | ≈ 0.011 | 0.0003–0.0006 |

This model excludes the systematic effect of new embryos, which turned out to be the dominant term. Changes that add
divisions are about 3.8× noisier on the public leaderboard than edge-only changes.

## 4. The strict gate (S1–S7), from P16 onwards

`research/cl/strict.py <rows.json> <vi> [label]` checks the candidate `vi`, an index into the `rule_eval.py` rows where
0 is the base:

| Id | Check |
|---|---|
| S1 | the delta over all 199 movies is > 0 and its bootstrap CI lower bound is > 0 |
| S2 | each embryo Δ > 0 with bootstrap P(Δ > 0) ≥ 0.95 |
| S3 | clean40 Δ > 0 |
| S4 | B5-unseen72 Δ ≥ 0 |
| S5 | leave-one-group-out: Δ > 0 with any one of the five groups removed |
| S6 | drop-top-5 Δ > 0; per embryo drop-top-3 Δ ≥ 0 |
| S7 | more movies better than worse (S7b: counting only movies whose edges or forks changed) |

The strict gate has **blind spots for deletion rules**:

- Deleting any node shrinks the node-count penalty, which inflates S1 and S7.
- S6 ranks movies by a per-movie contribution in which a false fork weighs only 0.2 of an edge, so deleting false forks
  barely registers.
- A movie bootstrap cannot turn negative for a rule that only deletes forks.

So from P19 on, candidates also had to pass an **8-point multi-check**:

1. an **independent re-implementation**, plus replay on the submission CSV and on Kaggle;
2. a **matched placebo** with at least 100 draws: the same number of changes, made at random or in random directions. The
   gain must come from the specific change, not from moving things;
3. **every one-step parameter neighbour** must pass;
4. **ablation and leave-one-rule-out;**
5. **cross-base**: the same rules applied to other base graphs (B5, P13, P14, P15, P17) must agree in sign;
6. exact **drop-top-k**;
7. a **node / edge / division split** with event counts;
8. **thresholds chosen on one subset and evaluated on another.** Never pick a threshold from a table of all-data GT
   outcomes.

These checks come on top of the per-embryo, clean40 and B5-unseen72 breakdowns that every comparison reports.

Several of these reviews were run as independent, adversarial passes. Each pass tried to refute the claim, and the verdicts
were aggregated as `SAFE`, `WEAK` or `REJECT`.

## 5. Leave-one-embryo-out (LOEO)

The hidden test set is embryo-disjoint, so every *learned* component was re-trained on one embryo and evaluated on the other
(`research/cl/loeo.py`):

| Linker model | Gain on clean40 | Gain on t127 + audit32 |
|---|---:|---:|
| deployed (trained on t127 + audit32) | +0.00386 | +0.00950 (in-sample) |
| cross-embryo (trained on the other embryo only) | +0.00355 | +0.00378 |
| in-embryo 5-fold | +0.00346 | +0.00419 |

That suggested about 90 % transfer for the linkers. In practice they transferred much less (see
[`PUBLIC_VS_PRIVATE.md`](PUBLIC_VS_PRIVATE.md)).

LOEO killed several things that looked excellent within one embryo:

- **junk-track pruning**: −0.40 when trained on 44b6 and applied to 6bba, −0.11 even with a 10 %-per-movie deletion cap;
- a retrained division scorer;
- image-based link re-scoring.

## 6. Pre-registration

For risky or bet-like changes I wrote the acceptance rule *before* seeing the result. An example is
`research/cl/p21/PREREG_P21.txt`, the division-threshold bet:

- required: division precision ≥ 43 % on newly evaluable forks, clean40 ≥ 0, a non-negative half-step neighbour, and the
  same sign on a second base;
- the bet failed the pre-registered rule;
- it was then submitted anyway as an explicitly labelled bet. It was +0.001 public and +0.001 private.

## 7. Kaggle-side verification

Before every submission:

1. **Build the notebook from the last verified one.**
   - From P21 on, a generator changed only the config name, the variant label and the description, and a diff guard asserted
     that nothing else changed. `notebooks/build_variant.py` is the cleaned-up version.
   - Earlier variants used `research/cl/build_pkernel.py`, which also switched the stage script and the reported status
     fields.
2. **Run it on Kaggle** (the "Save & Run" preview on the 4 visible test movies).
3. **Check the output.**
   - `research/cl/verify_kout.py` checks the variant, the 16-hex-character SHA-256 prefix of every stage file, zero
     whole-movie and DSR errors, and the runtime. I also scanned the per-movie records for `_error` or `_skipped` entries.
   - The preview `submission.csv` is scored locally with the official metric and must match the local pipeline to within
     floating-point noise (for example Kaggle 0.951690 vs local 0.951692).
   - Node counts may differ by 1–4 per movie (T4 vs H100 arithmetic); edges and forks must agree.
4. **Compare byte for byte.** Where two variants should produce identical output on the preview movies, the CSVs had to be
   byte-identical. For example, P23A (P23C plus jump relinking, not submitted) differs from P23C only in the jump step, and
   none of the preview movies has a jump.

## 8. Reading the public leaderboard

- **Rounding.** The public leaderboard shows 3 decimals, and most late changes were < 0.001. I compared "public minus local"
  gaps rather than raw public scores.
- **Unrounded ordering.** When exactly one submission is manually selected, Kaggle's *auto-selection* marker points at the
  best **unrounded** public score among the remaining ones. Toggling selections therefore orders submissions that all
  display the same rounded score.
- **Pre-set decision rules.** From the P11 submission onwards, I wrote down before each submission which result would trigger
  which follow-up.

## 9. Where the protocol failed

The protocol was good at rejecting learned deletions and division *replacements*: the DivNet variants it warned about were
the worst P-series submissions on private (0.932–0.933). It failed on division *recall*:

- **Local data under-rates extra divisions.**
  - The start-threshold band [0.80, 0.85) was rejected locally (0 TP / 5 FP on evaluable forks), yet P23C, which used it,
    was my best late private score (0.938).
  - `shortbranch` was rated fragile, yet P17, which kept it, was not worse on private than P15 and P20 (0.937 vs 0.936,
    one rounding step).
  - On b1-trained movies, true divisions score very high, so the low-score bands look like pure false positives.
- **It trusted the public leaderboard on division components.** DSR passed locally on both embryos and on clean40, and I
  removed it because the public score dropped by 0.002–0.003. With about 40 annotated divisions in the public part, that is one
  or two events. On private it was worth +0.004, second only to start-type completion.
- **Its transfer estimates for edge rules were too optimistic.** LOEO said about 90 % of the +0.003 linker gain would transfer;
  the private leaderboard shows no visible gain (< 0.001). A second embryo is not a good enough stand-in for an unknown third
  one.
