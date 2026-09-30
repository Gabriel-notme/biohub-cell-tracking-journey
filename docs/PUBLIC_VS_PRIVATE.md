# Post-mortem: public #2 → private #36

At the deadline my best public score was **0.976**, second on the public leaderboard; the leader had 0.978. On the private
leaderboard my selected pair (P21 + P20) scored **0.937**, rank **36 of 3,995**, a drop of 34 places. Three of my
*unselected* submissions scored **0.939**.

Kaggle scores every code submission on the whole hidden test set. I recorded both the public and the private score for 28
submissions, and many of them differ from another submission by exactly one component. This page uses those pairs as a
controlled experiment.

All leaderboard numbers are rounded to 3 decimals. A delta of 0.000 means |Δ| < 0.001, and ±0.001 deltas are at the
resolution limit. The complete table is in [`results/submissions.csv`](../results/submissions.csv).

## 1. The overall picture

| Stage | Local clean40 | Public | Private |
|---|---:|---:|---:|
| B5 (frozen base) | 0.9667 | 0.965 | 0.928 |
| P1 (division completion) | 0.9711 | 0.971 | 0.934 |
| P7 (P1 + DSR + dfork + learned linking + prune order) | 0.9789 | 0.972 | **0.939** |
| P15 (no DSR; + edge rules) | ≈ 0.981 | 0.975 | 0.936 |
| P21 (P15 + clean-up + start threshold 0.85) | ≈ 0.981 | **0.976** | 0.937 |

"Local clean40" is my held-out score on 40 movies (36 held-out + the 4 preview movies); see
[`VALIDATION.md`](VALIDATION.md).

The absolute private level (~0.93–0.94 for everyone around me) is much lower than public (~0.97): the private embryos appear
to be harder. What matters here is the *ordering*:

| Rank correlation (Spearman) | Local clean40 vs private | Public vs private |
|---|---:|---:|
| 9 early models (B3, B5, P1–P7) | **0.88** | 0.63 |
| 13 models (adds P13, P14, P15, P17) | 0.68 | 0.35 |
| all 21 P-series submissions | n/a | **0.29** |
| all 28 submissions | n/a | 0.71 |

My local held-out score predicted the private order better than the public leaderboard did, but it was far from perfect.
For the late models it fails outright: P15 and P21 rank above P7 locally (≈ 0.981 vs 0.979) and below it on private
(0.936 / 0.937 vs 0.939).

## 2. Component-by-component

Each row compares two submissions that differ only in the named component.

- "Local" is the delta on clean40, except rows marked *(199)*, which use all 199 labelled movies.
- "n/a": DivNet was trained on every labelled movie, so it has no fair local number.
- "↓": the unrounded public score was *not* higher. I read the unrounded ordering from Kaggle's auto-selection marker.

| Component | Pair | Local | Public | Private |
|---|---|---:|---:|---:|
| start-type division completion | B5 → P2 | +0.0030 | +0.005 | **+0.007** |
| + stolen-type completion | P2 → P1 | +0.0014 | +0.001 | −0.001 |
| double-fork resolution K=2 | P1 → P4 | +0.0022 | +0.003 | +0.001 |
| **dropped-sister recovery (DSR)** | P4 → P3 | +0.0016 | **−0.002** | **+0.004** |
| same DSR forks, with learned linking | P6 → P5 | +0.0016 | **−0.003** | **+0.004** |
| learned relink + free-end linking | P4 → P6 | +0.0030 | +0.001 | 0.000 |
| same linkers, with DSR | P3 → P5 | +0.0030 | 0.000 | 0.000 |
| prune after linking | P5 → P7 | +0.0010 | 0.000 | 0.000 |
| prune order + whole-movie dfork | P6 → P13 | +0.0016 | 0.000 | 0.000 |
| term_trim + long_link | P13 → P14 | +0.0011 | 0.000 | 0.000 |
| tbext + relinefit | P14 → P15 | +0.0016 | 0.000 | +0.001 |
| cutdup + forkfrag + shortbranch | P15 → P17 | +0.0015 | −0.001 | +0.001 |
| robust clean-up (no shortbranch) | P15 → P20 | +0.0003 *(199)* | 0.000 | 0.000 |
| DivNet blend (replaces b1 ranking) | P15 → P16a | n/a | −0.002 | −0.004 |
| DivNet rerank | P15 → P16b | n/a | −0.008 | −0.004 |
| DivNet blend on top of clean-up | P17 → P18 | n/a | −0.002 | −0.004 |
| start threshold 0.90 → 0.85 | P20 → P21 | 0.0000 (199: +0.0003; B5-unseen: −0.0012) | +0.001 | +0.001 |
| start threshold 0.85 → 0.825 | P21 → P23D | ≈ −0.0007 *(199, estimated from P23B − P22B)* | 0.000 ↓ | 0.000 |
| start threshold 0.85 → 0.80 | P21 → P23C | −0.0007 (199: −0.0011) | −0.001 | +0.001 |
| global-jump relinking | P20 → P22A | +0.0001 *(199)* | 0.000 ↓ | 0.000 |
| global-jump relinking | P21 → P22B | +0.0001 *(199)* | 0.000 ↓ | 0.000 |

## 3. What the pairs say

### 3.1 Division recall carried most of the private gain

The two largest private gains both come from **finding more divisions with the b1 fork head** (b1 is the event model from
B1; see [`SOLUTION.md`](SOLUTION.md) §2.2):

- **Start-type completion, +0.007.** Here the public leaderboard agreed (+0.005).
- **DSR, +0.004.** Here the public leaderboard got the sign wrong (−0.002 / −0.003). The two rows are nested pairs: the
  same DSR forks, measured with and without the learned linkers. They are not two independent replications.

Lowering the start threshold further was at the resolution limit on private: +0.001 for 0.90 → 0.85, then 0.000 for
0.85 → 0.825 and +0.001 for 0.85 → 0.80. It did not hurt, which is more than my local data predicted.

Divisions carry 10 % of the score through a micro-averaged Jaccard over a small number of events. Two things made recall
look worse than it was:

- **Locally**, on movies b1 was trained on, true divisions get very high scores, so the low-score bands look like pure false
  positives (0 TP / 5 FP in [0.80, 0.85)).
- **On public**, with perhaps 40 annotated divisions, one or two unlucky events can flip the sign of a small component such
  as DSR.

A possible exception is **stolen-type** completion, which takes a daughter away from another track. Its b1 AUC dropped from
0.89 to 0.81 on unseen movies, and its single private reading is −0.001. Locally, though, it was positive everywhere:
clean40 +0.0014, and removing it cost 0.0031 on 199 movies, on both embryos. One −0.001 reading at the resolution limit is weak
evidence. Adding a *new* daughter may be safer than re-assigning an existing one, but that is not established.

### 3.2 Edge-level repairs transferred much less than any local estimate

The learned linkers were my best-validated component:

- +0.003 on clean40 with a CI clearly above zero;
- positive on both embryos;
- about 90 % retained under leave-one-embryo-out training.

On private they are invisible: 0.000 on both nested pairs, so the true effect is below 0.001. The same holds for the chain
of rule-based edge fixes from P6 to P20: about +0.004 on clean40, of which P13 → P15 alone is +0.0027, and at most +0.001 on
private.

Transfer from *one* other embryo was not a reliable estimate of transfer to *new* embryos. A likely reason is that motion
speed, density and detection quality differ between embryos, and these gains probably depend on the particular detector
failures that occur in the training embryos.

### 3.3 Replacing b1's division choices was bad everywhere

All three DivNet variants lost about 0.004 on private, in line with the public leaderboard and my cross-embryo tests. Here
local, public and private agreed, as they did for start-type completion and double-fork resolution.

### 3.4 What a different selection would have scored

| Selection | Private (best of the two) |
|---|---:|
| P21 + P20 (what I chose: best public + its division-neutral parent as a hedge) | 0.937 |
| P7 + P21 (one per hypothesis: best local division model + best public model) | **0.939** |
| P23C + P21 (both division-recall bets) | 0.938 |

## 4. Why I made the wrong call at the time

- **I read DSR's public drop as a mechanism, not as noise.** The story was that "weak-evidence additions do not transfer to
  new embryos". It was plausible, and it matched the failures of node-level additions (junk-track pruning, fragment
  re-insertion) that *had* failed cross-embryo tests. DSR, however, was positive on *each* embryo separately (44b6 +0.0018,
  6bba +0.0017). I let two public readings of −0.002 and −0.003, taken on the same forks, outweigh that.
- **I believed removing a component was the safe direction.** It is only safe if the component is neutral. In a metric
  where one division is worth +0.0016, removing a true-positive source is as risky as adding a false one.
- **The public leaderboard steered more and more decisions as the deadline approached.** It decided:
  - not submitting P19 (my rule required P17 ≥ 0.976);
  - building the last bets (P23A/B) on P22B rather than P22A, based on P21's public score;
  - dropping P23A/B for P23C/D, based on the unrounded public order;
  - the final pick, P21.
- **My hedge protected against the wrong risk.** P20 was chosen as a division-neutral fallback for P21. The two differ by
  about one fork per movie, so the second pick carried almost no independent information. The real uncertainty was
  DSR-style recall versus none, and neither pick contained DSR.

## 5. Takeaways I will reuse

1. **Count the evaluable events behind a public delta.** If a public change can be explained by one or two division events,
   it is not evidence.
2. **Keep the component that local held-out data supports on every split, unless the leaderboard evidence is large compared
   with its event noise.**
3. **Diversify the two final picks by hypothesis,** for example "aggressive divisions" vs "robust edges", not by score or by
   minimal diffs.
4. **Be suspicious of transfer estimates built from a second training embryo.** Treat them as upper bounds.
5. **Calibrate event detectors on the test distribution itself** (per-movie score quantiles, or self-consistency checks
   between the two daughters) rather than trusting thresholds tuned on in-sample movies.
