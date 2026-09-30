# Post-mortem: public #2 → private #36

At the deadline my best public score was **0.976**, second on the public leaderboard; the leader had 0.978. On the private
leaderboard my selected pair (P21 + P20) scored **0.937**, rank **36 of about 4,000**, a drop of 34 places. Three of my
*unselected* submissions scored **0.939**.

Kaggle scores every code submission on the whole hidden test set. I recorded both the public and the private score for 28
submissions, and many of them differ from another submission by exactly one component. Sections 1–3 use those pairs as a
controlled experiment.

Section 4 compares my result with the rest of the field. That comparison shows the final selection cost me only about a
dozen places. The larger limits were ones that sections 1–3 cannot measure: a detector I could not validate, and a public
leaderboard drawn from a different embryo than the private one.

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

The absolute private level (~0.93–0.94 for everyone around me) is much lower than public (~0.97). At first I read this as
"the private embryos are harder". That is only part of the story. The private embryo does differ from the public one, and
most teams near me dropped by 0.02–0.04. But the winner scored 0.977 on private, slightly *higher* than on public, and my
drop was the largest in the private top 49 (section 4). Within my own submissions, what matters is the *ordering*:

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

| Selection | Private (best of the two) | Private rank |
|---|---:|---:|
| P21 + P20 (what I chose: best public + its division-neutral parent as a hedge) | 0.937 | 36 |
| P7 + P21 (one per hypothesis: best local division model + best public model) | **0.939** | about 22–26 |
| P23C + P21 (both division-recall bets) | 0.938 | about 26–33 |

The rank ranges come from ties: four other teams show 0.939 and seven show 0.938, and I cannot see the unrounded order.

## 4. How the rest of the field fared

The private leaderboard is still marked preliminary, so these ranks may shift slightly. "Public" is each team's best
public score, which is not necessarily the submission it selected.

| Private rank | Team | Public rank | Public | Private | Change |
|---:|---|---:|---:|---:|---:|
| 1 | Sergio Alvarez | 3 | 0.976 | 0.977 | +0.001 |
| 2 | Soheil Ayati | 28 | 0.968 | 0.970 | +0.002 |
| 3 | yu4u | 1 | 0.978 | 0.967 | −0.011 |
| 4 | Barry | 20 | 0.969 | 0.962 | −0.007 |
| 5 | Tang | 8 | 0.971 | 0.954 | −0.017 |
| 6 | Cyrus | 94 | 0.962 | 0.953 | −0.009 |
| 7 | tatsutaka | 7 | 0.971 | 0.952 | −0.019 |
| 12 | Corwin | 14 | 0.970 | 0.946 | −0.024 |
| 18 | ymg_aq | 9 | 0.971 | 0.941 | −0.030 |
| **36** | **Gabriel (me)** | **2** | **0.976** | **0.937** | **−0.039** |

- **Prize and medal lines.** Seven places were paid, and 7th place scored 0.952. The competition page lists 4,017 teams, so
  Kaggle's medal formula ends the gold zone at about rank 18, which scored 0.941. Rank 36 is in the silver range.
- **A better selection would not have changed the medal.** My best submission (0.939) would have tied for about 22nd–26th,
  still silver. The selection mistakes in section 5 cost about a dozen places.
- **Most of the field dropped, and my drop was the largest near the top.** Among the private top 49:
  - the top two gained +0.001 and +0.002, and 3rd place dropped 0.011;
  - ranks 4–11 dropped by 0.002–0.019;
  - most other teams in ranks 12–49 dropped by 0.020–0.036;
  - my −0.039 was the largest of the 49.
- **The public and private parts were different embryos.** The 3rd-place team inferred by probing that the public part
  (about 60 movies) and the private part (about 106 movies) each come from one new embryo. By their density measure the
  public embryo is dense, while the private one is much sparser. The public leaderboard therefore measured a different kind
  of data from the private one, not just a smaller sample of it. Their probed counts (about 166 hidden movies) are also
  lower than the ~199 movies (58 public + 141 private) I assumed from the data page in [`SOLUTION.md`](SOLUTION.md) §1
  and in the noise model of [`VALIDATION.md`](VALIDATION.md) §3.
- **Most of my drop was there from day one.** The public notebook I built on scored 0.947 public and 0.916 private, a drop
  of 0.031. Eight teams with a public score of 0.962–0.963 finished at 0.938–0.939 private, level with my best
  submissions; my public lead over them did not survive.

My own work was real on private: from 0.916 for the base notebook to 0.939 for P7. But after P7, the extra +0.004 public
that P21 gained came with −0.002 private.

### 4.1 What the published write-ups say

These summaries come from the write-ups posted after the deadline. The winner's write-up had not been posted when I wrote
this.

**Teams that trained their own detectors:**

- **2nd place** ([write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/2nd-place-solution),
  0.968 public / 0.970 private):
  - trained its own temporal 3D U-Net detectors, five folds per model;
  - excluded regions that are neither annotated nor clearly background from the loss, so that missing annotations do not
    become negative examples;
  - added synthetic faint cells as augmentation.
- **3rd place** ([write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/3rd-place-solution),
  0.977 / 0.967 for its selected submission, best public 0.978): ensembles of its own five-fold 2.5D and 3D detectors. It still dropped about 0.01, and its probing is the
  source of the embryo split above.
- **4th place** ([write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/4th-place-solution),
  0.969 / 0.962):
  - trained its own detector, with fold copies for validation, and checked detector changes across embryos;
  - deliberately left out the public heatmap detector, because its weights had seen all 199 training movies and could not
    be cross-validated;
  - reports that private stayed close to public across all of its submissions.
- **5th place** ([write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/5th-place-3d-u-net-transformer-linker-multi-s),
  0.971 / 0.954):
  - trained its own five-fold detector and linker, pretrained on public ZebraHub data;
  - credits the external pretraining with +0.016 private, the fold ensemble with +0.007 and test-time augmentation with
    +0.005.
- **16th place** ([write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/16th-place-solution),
  0.963 / 0.943): self-trained 3D U-Net detectors on top of the public tracking stack, and reports that its larger private
  gains came from detector changes.

**Teams that kept the public detection, as I did:**

- **12th place** ([write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/12th-place-solution),
  0.970 / 0.946), the route closest to mine:
  - tuned, but did not retrain, the same public detector weights, on a different public chain from mine;
  - built its own measured, fail-safe repair stages, including one that gives a single-child mother its second daughter;
  - reports that its in-sample bench over-read private by about 0.03, and that the stages it added after its base champion
    were worth about twice as much on private as on public (+0.013 vs +0.006), while its earlier consolidation stages
    gained less on private.
- **18th place** ([write-up](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/18th-place-solution-lineage-graph-refinement),
  0.971 / 0.941):
  - kept a public notebook's detection and candidate graph, and rebuilt the division decisions downstream;
  - their private score levelled off at about 0.942 from 20 September, while their public score kept rising to 0.971.

### 4.2 What this means for my result

- **The detector I could not validate limited everything downstream.** It was trained on all 199 movies, so none of my
  validation groups was clean for detection ([`VALIDATION.md`](VALIDATION.md) §2). I responded by freezing it and
  repairing its output. The prize-place write-ups I read (2nd–5th) all describe their own detectors, with fold copies they
  could validate.
- **The public detector did not decide the medal by itself.** The public-detection pipelines I can check finished at
  0.937 (me; 0.939 for my best, unselected submission), 0.941 (18th) and 0.946 (12th, inside the gold zone). The
  12th-place team reached 0.946 with the same detector weights, tuned but not retrained, on a different public chain plus
  its own stages. That is about 0.007 above my best, a rough gap given about 0.003 of read noise on each score, but it
  shows that more was possible from a comparable foundation. A self-trained detector was not enough on its own either:
  the 16th-place team had one and finished at 0.943.
- **My public tuning was aimed at the wrong embryo.** The public part is a single dense embryo and the private part a
  sparser one, so my late public-guided choices (section 5) were in effect tuned on the dense one. That fits the pattern
  after P7: +0.004 public, −0.002 private.
- **The selection itself was the smallest problem.** Choosing P21 + P20 instead of a pair with P7 cost about 0.002 private
  (0.939 vs 0.937), about a dozen places.
- **Some of my local signals may have pointed the wrong way.** b1 was trained with hard negatives mined from the real
  detector output. With labels this sparse, many of the hardest "negatives" may be real, unannotated divisions. The
  4th-place team reports that hard-negative mining hurt their division models for exactly this reason. It would also
  explain why the changes that add a new daughter (start-type completion, DSR, lower start thresholds) were positive or
  neutral on private, while my local data called the low-score bands false positives. This is a hypothesis; I have not
  tested it.

## 5. Why I made the wrong call at the time

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

## 6. Takeaways I will reuse

1. **Build the foundation from parts I can validate.** If a public model was trained on all the labelled data, retrain an
   equivalent per fold or per group instead of freezing it and optimising around it. Put the effort into the main model,
   external pretraining, fold ensembles and test-time augmentation before post-processing.
2. **Find out what the public split actually is.** If the public part can come from a different group (here, a different
   embryo) than the private part, public gains are gains on that group only.
3. **Count the evaluable events behind a public delta.** If a public change can be explained by one or two division events,
   it is not evidence.
4. **Keep the component that local held-out data supports on every split, unless the leaderboard evidence is large compared
   with its event noise.**
5. **Diversify the two final picks by hypothesis,** for example "aggressive divisions" vs "robust edges", not by score or by
   minimal diffs.
6. **Be suspicious of transfer estimates built from a second training embryo.** Treat them as upper bounds.
7. **Calibrate event detectors on the test distribution itself** (per-movie score quantiles, or self-consistency checks
   between the two daughters) rather than trusting thresholds tuned on in-sample movies.
8. **Compare with the field, not only with my own submissions.** I explained my absolute drop as "harder test data", but it
   was much smaller for the leaders: the top two did not drop at all, and 3rd–7th place dropped 0.007–0.019, against my
   0.039.
