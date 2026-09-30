# Experiment log: what did not work

Most entries here were measured with the official metric on full movies. Entries marked **(AUC)** were rejected on
classifier AUC or candidate precision alone, without a full-pipeline run. I keep this list because most of these ideas are
the natural next thing to try, and each one cost real GPU hours.

Abbreviations:

- **clean40** = hold36 + prev4, the held-out movies;
- **LOEO** = leave-one-embryo-out (train on one embryo, test on the other);
- **evaluable fork** = a fork that lands on an annotated lineage;
- **break-even** = the precision, or the number of annotated edges per node, at which a change's expected gain exactly
  equals its expected cost. The cost is the true edges or divisions it loses, or the node-count penalty of the nodes it adds
  or keeps;
- **b1 / b2** = the event-model networks from B1 / B2 (see [`SOLUTION.md`](SOLUTION.md) §2.2).

Script paths are under [`research/`](../research).

---

## 1. B-series (learned lineage models on the public base)

| Idea | Result |
|---|---|
| Track and image networks trained on **GT-centred** crops and tracks | Near-perfect classification AP, but whole-movie scores regressed: GT centres and clean tracks do not look like detector output |
| Cross-embryo image heads | Same failure; high AP, lower full-graph score |
| Large-scale edge replacement; endpoint-only repair; high-recall repair | Unstable between calibration and audit |
| XE-NDCG ranking loss | No change under the conservative decoder (the replacement gate that only swaps an edge when the old probability is ≤ 0.2 and the new ≥ 0.8) |
| Teacher-branch merge (merging a second, B3-based "teacher" lineage graph into the candidate set) | No gain |
| Whole-pipeline FP16 | Changed detections and post-processing; not equivalent |
| Removing D4 TTA for speed | Not equivalent; kept 8-view FP32 |
| 11-frame event model, photometric augmentation, coordinate jitter, low-LR continuation | No gain over the 3-frame b1 / 5-frame b2 |
| B5/B6 complexity (track-video transformer, set attention, LambdaRank) over B3/B4 | +0.001 calibration, +0.001 audit, **0.000 public**, 0.000 private |

## 2. P1–P7: division completion and linking

| Idea | Result |
|---|---|
| Link first, then complete divisions | prev4 gained a false division |
| Second division-completion pass | Worse |
| Re-insert dropped detection fragments (one-end attach) | prev4 −0.002; two-end attach not significant |
| Retrain the linker on the new graph distribution | Equal to P7 |
| Relink threshold 0.5; prune length 3 | Not significant |
| Image appearance "is this a real cell" classifier **(AUC)** | AUC 0.72; cannot delete false detections or recover real ones |
| b2 division head, or b1 + b2 ensemble | Does not see the missed divisions |
| Dense-transformer edge probabilities as a second opinion **(AUC)** | Relink/swap AUC 0.71 (negative expected value); free-end AUC 0.909 → 0.945 but about +2.5 % of the gain; divisions AUC 0.60–0.66 |
| Arbitrating B5's own lineage-stage changes | 75–97 % of them are correct; undoing any loses |
| GBM division completion from b1 + track-structure features | After fixing mislabelled training candidates, no high-precision candidates; hold36 lost 4 true divisions |
| Delete false forks by structural features | Every feature AUC 0.4–0.6, with the direction flipping between train and validation |
| Learned identity-swap repair **(AUC)** | AUC 0.76; negative expected gain in training CV |
| Fill holes from dropped detections; extend track ends | AUC 0.67–0.75; the node cost cancels the gain |
| Off-position node correction | Only 14 cases, 7–9 µm off |
| b1 image features in the linker | AUC 0.938 → 0.942; negligible |
| Add hold36 or prev4 to linker training | Worse on the set that was held out (prev4 −0.0009) |

## 3. P8–P13: node deletion, divisions, and the first cross-embryo tests (P8 = P6 with pruning after linking; not submitted)

| Idea | Result |
|---|---|
| **Junk-track pruning** (Poisson GBM predicting annotated edges per track; `jprune.py`) | clean40 +0.0009, but LOEO 44b6 → 6bba **−0.40**; with a 10 %-per-movie cap −0.11; with a relative threshold −0.019. Whether a track is annotated flips between embryos (isolated tracks are rarely annotated in 44b6 and often in 6bba) |
| Node deletion by length or component size | ≤ +0.00006; tracks ≥ 6 nodes carry 2.7× the break-even number of edges |
| Extend ends/starts without image evidence | +1 frame adds 15 k nodes, +9 TP / +39 FP edges |
| Prune low-confidence edges | Even the lowest band is 65 % correct; the break-even is about 48 % |
| Replace nodes with a closer dropped detection **(AUC)** | Graph-feature AUC 0.68; image-feature AUC 0.47–0.55 |
| Re-insert dropped detection chains as tracks | 0.0023 true edges per node; the break-even is 0.004 |
| "Annotation is late" timing hypothesis | Only 4 missed divisions are a few frames off |
| Anatomy features of the q-track (the track the second daughter would be taken from) for stolen divisions | TP : FP about 1 : 3,000; no separation |
| b1 fork head with D4 TTA | Too few labelled candidates to validate; among 199 movies, the forks it demotes are half right |
| Lower division thresholds (0.8–0.9 band) | About 20 % precision locally, below the ~26–30 % break-even. *Private later disagreed (see PUBLIC_VS_PRIVATE.md)* |
| Unrounded coordinates | 0.979830 vs 0.979898 rounded |
| Image features in junk pruning | +0.0001 in CV |
| GT division regularities (sister distance, division time, parent track length) | True and false forks have the same distributions |
| Daughters living ≤ 3 frames | 1 true, 2 false; too few |
| Prune thresholds 3–6 nodes | k=3 +0.00008; k=6 negative on clean40 |
| Retrain the linker on the new intermediate graphs | clean40 +0.0003, 44b6 only |
| "Early split" geometry rule (a new track appears next to an existing one, then separates) | 691 forks added on 199 movies, 0–2 true |
| Global detection offset | Bias < 0.5 µm and opposite between embryos |
| b1 edge features in relinking **(AUC)** | Cross-embryo AUC +0.01–0.02; not worth deploying |
| Disabling stolen-type completion | −0.0031 on 199 movies, both embryos negative (kept). *A single private reading of stolen-type was −0.001, at the resolution limit* |
| Border-node deletion | Border annotation density is 1/3–1/2 of the interior, still 3.7–5× break-even |
| b1 veto of suspicious B5 forks | 3,277 forks, almost all b1 ≥ 0.9; true and false are indistinguishable |
| Longer gap bridging (3–6 frames) | 27 interpolable gaps; ceiling +0.0008 |
| B3 as a second opinion | About 7 h extra; over the 12 h limit |

**Virtual parent** (`vparent.py`, `ft_*.py`). b1 is trained on crops centred on the annotated parent. When the detector
pre-splits a dividing nucleus, the "parent" crop is off-centre and b1 scores near 0. Re-cropping at the p/q midpoint raised
the AUC on the missed stolen-type divisions (those with a pre-split parent) in unseen movies from 0.694 to 0.752, and from 0.885 to 0.927 under LOEO. However, among *genuinely new*
divisions at b1 ≥ 0.97, precision was 3.5 % (3 TP / 82 FP), and fine-tuning b1 in this input form was unstable.

## 4. P13–P15: oracle bounds and rule families

**Oracle bounds (GT-aware, 199 movies).**

- Perfect relink + gap decisions: +0.0066.
- About 1,289 missed edges come from two predicted tracks straddling one annotated cell. Knowing which one is annotated
  would give +0.006–0.010; without that knowledge, deletion loses.
- Only 96 of the 274 k stolen-type candidates are true divisions.
- B5's own forks used as pseudo-labels are 55 % correct (29 % in 44b6).
- **Conclusion:** with the B5 detector frozen, post-processing cannot add +0.01.

| Idea | Script | Result |
|---|---|---|
| Parallel shadow-track deletion | `shadow.py` | D=4 µm +0.00006; D ≥ 5 µm negative on both embryos |
| Temporal coordinate smoothing | `smooth.py` | One embryo up, one down |
| GT maximum edge length cut | `len_inv.py` | No predicted edge exceeds it |
| Depth / time "unannotated region" priors | `zcov.py` | Different between embryos |
| Extended relinking beyond the candidate graph | `xrl*.py` | Every threshold negative under LOEO |
| Gap 3–7 bridging | `gap_pot13.py` | About 60 recoverable edges in total |
| GT geometric range of forks | `fork_prior13.py` | P13 true and false forks both inside it |
| Second round of rule families | `ideas/r2_*`, `r3_*` | < +0.0002 and single-embryo; duplicate ceiling +0.0003; gap ceiling +0.0012 (best GT-free +0.00001); fork-structure deletion removes 1 false fork |
| tbext at pmin 0.5 instead of 0.8 | `ideas/r3_tbext.py` | +0.00166 vs +0.00118, but it adds nodes on weak evidence; not used. Later, on P20: 0.4–0.6 gave CI including 0, clean40 negative |

## 5. P16 search under the strict gate (S1–S7)

| Direction | Result |
|---|---|
| B5 output-filter audit, 4 rule families × 2 independent reviews | cutdup +0.00012 (fails S2 / S6); ILP makes almost no forks; restoring capped "safe" divisions is 10 % correct; motion relink replacements are right 92 % of the time; forkfrag +0.00008 |
| Short-track filter family | Best +0.00030 (CI > 0) but 44b6 evidence comes from 2 movies; fails S2 / S6 |
| Smoothing the tbext-added nodes | −0.00010 |
| B5 lineage-stage ablation (GPU rerun, bit-exact) | Every stage helps or is neutral. On audit32, removing a stage gives: visual linking −0.0015, verified recovery −0.0042, motion joint optimisation −0.0053, fast division review −0.0086. Track ensemble: +0.00005 to +0.00042, noise level |
| Learned node-addition verifier | +0.00006 (ceiling +0.0017, 537 evaluable samples) |
| Learned edge re-scoring incl. swaps and b1 features | ≈ 0; positives are 0.2–0.3 % of candidates |
| Learned coordinate refinement (3D CNN) | −0.0007 to −0.011; the +0.024 ceiling is z / annotation noise with no image signal |
| Learned division completion | +0.0006, all from b1-trained movies; clean40 negative |
| Global re-solve (learned costs + ILP) | −0.0004; with GT costs, a global solve adds 0 over local repairs |
| b1/b2 8-view flip TTA inside B5 lineage | 96 of 850 k edges change; score +0.00000; 5× slower |
| b1 flip TTA inside P-stage completion | −0.00091, clean40 −0.00235 |

## 6. P16–P20: upstream models and clean-up

| Idea | Result |
|---|---|
| **DivNet** (event-centred 3D CNN division scorer; `p16/divnet*`) | Better than a b1 retrained the same way under LOEO (+0.0041, clean40 +0.0107); but replacing or blending the *deployed* b1 −0.005 to −0.008; veto of false forks −0.0052. Public P16b **0.967**, P16a 0.973; private 0.932 |
| DivNet v2 (pseudo-label pretraining, hard negatives) | Each of the three variants failed 3–4 of the 5 pre-registered checks (M1–M5, see `research/cl/p16/divnet2/NOTES.md`) |
| Appearance link CNN (`p16/linknet`) | AUC above b1 but below the geometric features; no new information |
| b1 5-seed ensemble / 2× data (`p16/b1ens`) | Pre-registered checks: Q1 (5 seeds beat 1 seed) −0.0012, Q2 (doubled data) +0.0005 (P 0.59); b1's weakness is the embryo, not seeds or data |
| Two-model agreement for new divisions | 0 true divisions |
| True vs false fork structure (`p16/deploy/fork_an.py`) | Nearly identical; only "daughter disappears within 3 frames" separates, which became `shortbranch` |
| shortbranch deletion | +0.00042 from 2 false forks in hold36; maxk=3 gives −0.0014; deletes real forks on other bases. Dropped from P20 (on private, P17 was not worse: 0.937 vs 0.936, one rounding step) |
| b1 veto of existing P15 forks | 24 of 28 false forks score ≥ 0.9 |
| b2 veto of P-stage-added edges (θ 0.1–0.8) | −0.0002 to −0.0054 |
| b1 + b2 joint division score | −0.00247 (TP −1, FP +9) |

## 7. P21–P23: B5-level knobs and last-day ideas

These use a harness that reruns the Kaggle notebook's base pipeline on cloud GPUs. On the 99 movies checked, it reproduced
the original B5 scores to 5 decimals on every subset.

| Idea | Result |
|---|---|
| ILP division weight 1.2 → 0.4 (a forum tip from another team) | B5 level −0.0015 on every subset; after the P-stage +0.0012 with CI [−0.0008, +0.0046], all from one division in 2 movies; edges worse (TP −19 / FP +15), +3,125 nodes |
| ILP division weight 2.0 | Identical output: the ILP barely makes divisions at 1.2 |
| Detection threshold 0.95 / 0.94 | −0.0018 / −0.0001 after the P-stage; 44b6 and clean40 negative; +37 k / +54 k nodes |
| tbext threshold 0.5 / 0.6 / 0.4 on P20 | +0.0002 / +0.00025 / +0.0001, CI include 0, clean40 negative |
| Division timing shift | 74 of 84 missed divisions have no predicted fork nearby: a recall problem, not a timing problem |
| Depth-dependent z bias | 44b6 −0.12 µm, 6bba +0.45 µm, depth-independent; likely channel registration, not learnable |
| Frozen frames (947 pixel-identical transitions in 114 movies) | Already handled correctly |
| Global-jump relinking | +0.0001 locally, robust; no public or private gain |
| Local-flow, phase-correlation and probability relinking at jumps | −0.0001 to −0.0029, or ≈ 0 |
| Foreground node pruning with a nucleus-foreground U-Net (SimView) | In 44b6, GT-matched nodes get *low* foreground more often; rejected |
| Multi-hypothesis detection (naive version = lower detection threshold) | Negative (above); the principled version needs ILP node costs and exclusion constraints; left for future work |
