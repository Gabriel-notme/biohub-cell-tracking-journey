# Pipeline reference

This document specifies the submitted pipeline in enough detail to re-implement it. The final selected notebooks are
[`notebooks/p21-final-selection`](../notebooks/p21-final-selection) and [`notebooks/p20-final-selection`](../notebooks/p20-final-selection).
The best-private notebook is [`notebooks/p07-private-best`](../notebooks/p07-private-best).

Conventions:

- **Coordinates.** Voxel `(z, y, x)`, converted to µm with the scale `(1.625, 0.40625, 0.40625)`.
- **Track START / END.** A START is a node with no parent; an END is a node with no child.
- **"Fork".** A node with two children.
- **b1.** The event-model network trained for B1 and shipped in the B5 bundle (`b1_best.pt`). Its fork head scores
  (parent, daughter, daughter) triples.
- **Movie groups.** hold36, prev4, audit32, t127a and t127b; clean40 = hold36 + prev4 is the cleanest held-out set. See
  [`VALIDATION.md`](VALIDATION.md) §2.
- **Evidence columns.**
  - "local" is the official metric on my labelled movies (see [`VALIDATION.md`](VALIDATION.md));
  - "public" and "private" are leaderboard deltas between submissions that differ only by that step. Leaderboard values are
    rounded to 3 decimals, so a delta of 0.000 means |Δ| < 0.001.

---

## 1. Notebook layout (5 cells)

| Cell | Role |
|---|---|
| 0 | Markdown description of the variant. |
| 1 | Locate and hash-verify the frozen B5 artefact bundle (my public dataset `biohub-b5-b6-track-video-models-20260923`). It asserts the manifest SHA-256 and every model-file hash, and disables the baseline's internal event and validator hooks. |
| 2 | The public baseline, extended: dependency set-up from offline wheels, detection, association, ILP and post-processing. It also saves the **pre-ILP candidate graph** to `/kaggle/working/fullgraphs/*.geff` and the pre-lineage **reference graph** to `/kaggle/working/reference_graphs/`, and streams finished movies into my B5 lineage stages as a subprocess. |
| 3 | Waits for the streamed lineage subprocess (`streaming_pipeline.py` from the bundle, started in cell 2), which writes `lineage_graphs/` and the base `submission.csv`, then stamps `lineage_runtime_validation.json`. |
| 4 | **P-stage.** Copies `submission.csv` to `submission_base.csv`, runs `p_stage12.py` with the variant's config if under 10 h have been used, validates the new CSV, and restores the base CSV on any failure. Writes `pstage_status.json` with the aggregate counters of `pstage_report.json` and a 16-hex-character SHA-256 prefix of every stage file. P17 and P19 counters exist only in the per-movie records. |

The submission validator in cell 4 checks:

- the column order and a contiguous `id`;
- movie coverage;
- node-id uniqueness and non-negative coordinates;
- that every edge joins frames `t → t+1`;
- in-degree ≤ 1, out-degree ≤ 2, and no duplicate edges.

## 2. Public baseline (cell 2)

The baseline is the public notebook by zhincez, built on the public datasets by pilkwang. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The preset is `harmonic_v3_division_wide`. The knobs that matter:

| Group | Setting |
|---|---|
| Detection | temporal 3D U-Net, primary + secondary seed (`SECONDARY_DETECTION_WEIGHT 0.80`), `DET_THRESHOLD 0.965`, D4 8-view TTA, FP32 batched (`fp32_batched_d4_4_v1`) |
| Association | edge-feature TTA on both seeds; harmonic-probability bidirectional fusion (weight 0.15); secondary-seed low-margin consensus (≤ 0.35) |
| ILP | appearance 0.0, disappearance 2, division 1.2 |
| DeepCenter veto | TTA on; gap-repair threshold 0.25 (min span 8.5 µm); safe-division threshold 0.20 |
| Motion relink | tight 6 µm, relaxed 10 µm, learned bonus 1.0 |
| Gap repair | single-frame gap close ≤ 5 µm (reuse 3.2 µm); strict gap-2 recovery (step ≤ 4.4 µm, density-adaptive) |
| Safe divisions | daughters diverge ≥ 2.25 µm; parent–daughter ≤ 9 µm; sisters ≤ 14 µm; symmetry τ 0.6; global cap 0.375 % and per-frame cap 0.76 % of nodes |
| Output filter | drop fork-free tracks < 6 nodes; keep division components; short-track rescue (≥ 4 nodes, mean edge prob ≥ 0.88, mean step ≤ 3 µm) |
| Smoothing | ±2-frame line fit, weight 0.8, on fork-free chains |

Two baseline behaviours matter for the P-stage:

- **The output filter and motion relinking run after the ILP.** They throw away ILP-supported heads, tails and links, and
  drop every track shorter than 6 nodes.
- **Smoothing runs on the reference structure,** before any later re-wiring.

Both leave recoverable information in the saved candidate graph.

## 3. B5 lineage stages (cells 2–3, frozen)

Code: [`lineage_models/`](../lineage_models). `final_selection.json` in the bundle freezes the B5 variant as a base model plus
an ordered list of stages. Each stage has its own model files, thresholds and replacement gates:

1. **Event model (b1).** A 3-frame residual 3D encoder over local crops, with two heads:
   - a *fork head*: parent, daughter, daughter → P(division);
   - an *edge head*: source, target → P(same cell).

   It was trained on 159 movies with hard negatives mined from the real detector output (`b12/train_events.py`,
   `cell_event.py`, `refine_events.py`). The P-stage reuses this fork head.
2. **B3 stages:**
   - learned division review (`fast_division_review.py`);
   - rotation-invariant motion association (`motion_refine.py`, `fast_motion_features.py`);
   - 7-frame missed-cell recovery, confirmed by an independent verifier (`recover_joint.py`, `verified_recovery.py`);
   - 7-frame centre-offset refinement (`centroid_refine.py`).
3. **B4 visual edge review.** Image embedding + motion (`visual_edge_refine.py`), with forks protected.
4. **B5 track-video heads.**
   - The encoder `TrackRobust_137` combines local 3-frame 3D crops with a 10-step trajectory transformer, trained with
     corrupted-track augmentation (`track_video.py`, `train_track_video.py`, `prepare_corrupted_tracks.py`).
   - Two candidate-set attention heads (`TrackSet3_137`, `TrackSet3_823`; `track_set.py`) and a 3-seed LambdaRank head
     (`TrackRankRobustMatched`; `train_track_rank_robust_matched.py`) form a consensus.
   - A conservative replacement gate applies: the old edge probability must be ≤ 0.2 and the new ≥ 0.8, and the
     neighbourhoods of divisions are protected.

## 4. The P-stage (`pstage/p_stage12.py`)

**Inputs, per movie:**

- the B5 lineage graph (`lineage_graphs/<movie>.json`, passed as `--graphs`);
- the reference graph. There is no flag for it: the script reads `<graphs>/../reference_graphs/<movie>.json`, so the two
  folders must sit side by side, as in `/kaggle/working`;
- the pre-ILP candidate graph (`fullgraphs/<movie>.geff`, `--full`), holding detection ids, coordinates, candidate edges and
  `edge_prob`;
- the image (`<movie>.zarr` in `--data`, used for b1 crops);
- the B5 artefact bundle (`--artifact`), which provides `refine_events.EventRefiner`, `cell_event` and `b1_best.pt`.

**Execution.** Movies are sharded by graph size over at most two GPUs. Every optional step is wrapped in a `try` block and
followed by `check_graph`:

- no duplicate edges, no merges, out-degree ≤ 2;
- edges only join `t → t+1`;
- the frame set is unchanged.

A failure in one of these steps keeps the graph from before that step, and shows up as `<step>_error` in the movie's record.
Division completion and the pre-link prune have no per-step fallback: an error there, or in the final structural assertions,
keeps the whole movie's unmodified B5 graph and counts in the report's `errors`. A wall-clock deadline skips the remaining
movies. Export rounds coordinates to integers and clips them into the image box.

The step order below is the one in the final configuration, `p21s85_config.json`. DSR (§4.2) and jump relinking (§4.10) are
not in that configuration; they are described where they would run.

### 4.1 Division completion (`div_complete.py`)

**Candidates.** A parent `p` at frame `t` with exactly one child `a` (branch ≥ 2 nodes) is paired with a second daughter `b`
at `t+1`, where |p−b| ≤ 13 µm, |a−b| ≤ 20 µm and b's branch has ≥ 2 nodes. There are two types:

- **start**: `b` has no parent (it starts a new track);
- **stolen**: `b` is the only child of another parent `q`.

**Score.** The b1 fork head (`fork_ensemble=first`) scores (p, a, b) with trajectory fork-geometry features. Embeddings are
computed only for the candidate nodes.

**Accept.** Candidates are taken greedily by fork probability, each node used once:

- start if P ≥ `th_start`;
- stolen if P ≥ `th_stolen` = 0.97, detaching `b` from `q`.

| Evidence | Value |
|---|---|
| Local, hold36 (B5 → P1) | 0.96929 → 0.97485 |
| Ablations on clean40 (from P13) | − stolen −0.0037 (44b6 −0.0093, 6bba −0.0002); − start −0.0034 (44b6 0, 6bba −0.0057); − both −0.0072 |
| b1 fork AUC, b1-seen vs b1-unseen movies | start 0.965 → 0.973 (stable); stolen 0.891 → 0.814 (degrades) |
| Local precision of evaluable forks by band | start [0.85, 0.90): 2 TP / 4 FP; start [0.80, 0.85): 0 / 5; stolen [0.90, 0.97): 1 / 20 |
| Public | B5 0.965 → P2 (start only) 0.970 → P1 (+ stolen) 0.971; th_start 0.90 → 0.85: +0.001; 0.85 → 0.80: −0.001 |
| Private | B5 0.928 → P2 0.935 → P1 0.934; th_start 0.90 → 0.85: +0.001; 0.85 → 0.80: +0.001 |

A fork counts as "evaluable" when it lands on an annotated lineage. About 2–3 % of the added forks do.

`th_start` was 0.90 up to P20 and **0.85 in P21**. On public data a division TP is worth about +0.0016 and a FP about
−0.0007, so a band pays off above roughly 30 % precision.

### 4.2 Dropped-sister recovery, DSR (`dsr.py`; P3 / P5 / P7 only)

**Candidates.** For a parent with one child, *dropped* pre-ILP detections at `t+1` qualify when they are:

- more than 4 µm from every kept node;
- within 13 µm of the parent;
- within 20 µm of the existing daughter.

Each candidate is scored by the b1 fork head. The candidate's forward chain is used when it attaches (≤ 6 µm) to a free
START at `t+2`.

**Accept.** P ≥ 0.98, greedily. This adds the new node, the fork edge and, where available, the link to the future START.

| Evidence | Value |
|---|---|
| Local clean40 | +0.0016 (P4 → P3), both embryos positive (44b6 +0.0018, 6bba +0.0017); divisions +2 TP / +3 FP |
| Public | **−0.002** (P4 0.974 → P3 0.972), **−0.003** (P6 0.975 → P5 0.972) |
| Private | **+0.004** (P4 0.935 → P3 0.939; P6 0.935 → P5 0.939) |

I removed DSR after P7 because of the public signal. It was the costliest decision I let the public leaderboard make,
although my larger limits were upstream; see [`PUBLIC_VS_PRIVATE.md`](PUBLIC_VS_PRIVATE.md) §4.

### 4.3 Double-fork resolution (`dfork.py`)

If a fork has another fork within K frames in one of its branches, the earlier fork keeps only the branch that leads to the
later fork. Across all 199 labelled movies, none of the 151 annotated divisions has a daughter that divides again within the
movie; the cell cycle is roughly 550–630 frames.

- **K=2 (P3–P8).** Public +0.003 (P1 → P4), private +0.001.
- **K=100 (P11), then K=35 (P13+).** On 199 movies it removes 4 false divisions and 0 true ones; K=35 and K=100 give
  identical output on every labelled movie. K=35 protects a hypothetical fast-cycling embryo.

### 4.4 Learned relinking (`relink.py`, model `relink_lgb.json`, th 0.65)

**Candidates.** Candidate-graph edges `s → d` (dt = 1) that are absent from the final graph, where `s` and/or `d` are
currently linked elsewhere. The following are never touched:

- forks and fork daughters;
- fresh daughters;
- a current child that divides next.

**Features** (25): the candidate's pre-ILP edge probability, the probabilities and distances of the competing current links,
track history and future lengths, velocity-predicted position errors and direction cosines, z, relative time, and local
density.

**Accept.** Greedy by LightGBM probability. Accepting `s → d` removes s's current child link and d's current parent link.
Hold36 AUC is 0.73; the effect is small.

### 4.5 Learned free-end linking (`edge_link.py`, model `edge_lgb.json`, th 0.4)

**Candidates.** A track END at `t` is paired with a track START at:

- `t+1`, within 14 µm (gap 1);
- `t+2`, within 18 µm (gap 2). This inserts one node, snapped to a dropped detection within 3 µm of the midpoint, otherwise
  placed at the midpoint.

**Features** (26): the pre-ILP edge probability `fe`, distance and its components, track history and future lengths, speeds,
predicted-position errors, direction cosines, local densities, nearest-neighbour distances, the distance from the midpoint to
a dropped detection, and the candidate's rank and margin among competitors at both ends.

**Accept.** Greedy by probability, each end and start used once. Hold36 AUC is 0.938.

**Training (both linkers).** t127 + audit32 only, labelled by official edge matching; hold36 and prev4 were never used. The
thresholds come from the cross-validated expected gain. Deployment uses `lgb_np.py`, a NumPy evaluator of the JSON dump that
matches LightGBM to 2e-16.

| Evidence (linkers together) | Value |
|---|---|
| Local clean40 | +0.0030, CI [+0.0015, +0.0048] (P3 → P5 and P4 → P6) |
| Leave-one-embryo-out | trained on the other embryo +0.0036 vs +0.0039 deployed on clean40, i.e. about 90 % retained |
| Threshold scans | relink 0.5–1.01 and edge_link 0.3–0.6 all within −0.0004 … +0.0001 of the chosen values |
| Public | +0.001 (P4 → P6) |
| Private | 0.000 (P4 = P6 = 0.935; P3 = P5 = 0.939) |

### 4.6 Isolated-node pruning (`prune.py`, `post_prune: 2`)

Drop weakly connected components with fewer than 2 nodes.

P1–P6 pruned *before* linking. From P7 (and its unsubmitted no-DSR twin P8) it runs *after* linking, because many isolated
nodes are the missing frame of a broken track that the linker can re-attach. Local clean40 +0.001 (P6 → P8, almost all from
44b6); public and private 0.000 (P5 → P7, the pair that was submitted).

### 4.7 Duplicate-end trimming and long links (`p14_post.py`: term_trim, long_link; added in P14)

**term_trim** (r 3.5 µm, minlen 3, up to 5 iterations). A track END node lying within 3.5 µm of a same-frame node that
continues is treated as a duplicate detection and deleted:

- rounded submission coordinates are used for the distance;
- fork daughters are never touched;
- iterating removes whole duplicated tails.

If the continuing neighbour is a one-child START, the END inherits its continuation instead.

**long_link** (fe_min 0.5, min 14 µm). The linker only searches within 14 µm, but 6bba contains 120 GT edges longer than
that. Pre-ILP candidate edges END(t) → START(t+1) that are longer than 14 µm with probability ≥ 0.5 are added greedily. This
creates no nodes and no forks.

| Evidence | Value |
|---|---|
| Local, 199 movies | +0.00083, CI [+0.00051, +0.00118]; both embryos positive |
| term_trim alone | +0.00030; 154 movies better, 1 worse |
| long_link alone | +0.00053; new edges on annotated lineages 64 TP / 3 FP |
| Public / private | 0.000 / 0.000 |

### 4.8 Head/tail extension and re-smoothing (`p15_post.py`: tbext, relinefit; added in P15)

**tbext**, track head/tail extension (pmin 0.8, minlen 5, dup 3.5 µm, join=on). For every START or END of a track with ≥ 5
nodes:

- follow the pre-ILP candidate edge with probability ≥ 0.8 into a *dropped* detection of the adjacent frame, unless a kept
  node lies within 3.5 µm there;
- repeat frame by frame;
- if the added detection's own candidate neighbour on the far side is a free END or START, join and stop.

It never creates forks or merges. It restores ILP-supported heads and tails that the baseline's motion relink and 6-node
filter cut off.

**relinefit** (w 0.5). This redoes the baseline's ±2-frame line-fit smoothing on the *final* structure:

```text
new = cur + w * (fit_final − fit_ref)
fit = 0.2 * orig + 0.8 * linefit(orig over ±2 neighbourhood)
```

Here `orig` is the pre-smoothing coordinates. The weight is `w = 1 − centroid_blend` = 0.5, where `centroid_blend` = 0.5 is the
blend weight of B5's centre-offset refinement stage, which had already undone about half of the smoothing bias. About 7.7 %
of the nodes have a changed neighbourhood. The baseline's 2 µm collision rule is re-applied.

| Evidence | Value |
|---|---|
| Local, 199 movies (P14 → P15) | +0.00173, CI [+0.00112, +0.00237]; 44b6 +0.0014, 6bba +0.0018 |
| tbext alone | +0.00118 |
| relinefit alone | +0.00071 |
| relinefit placebo (same displacement, random direction) | −0.0014 |
| Public / private | 0.000 / +0.001 |

### 4.9 Deletion clean-up (`p17_post.py`, `p19_dup.py`, `p19_edge_deploy.py`)

The P20 configuration applies each of these only as a deletion:

- **cutdup.** Remove surviving baseline gap bridges (`gap_closed` / `gap2_recovered` synthetic nodes) that lie within 3.2 µm
  of another node of the same frame.
- **forkfrag.** Remove fork-free components of fewer than 6 nodes that contain a former B5 fork daughter which has lost its
  parent.
- **duplicate heads.** The START-side mirror of term_trim, at 2.5 µm.
- **term_trim re-run.** term_trim (section 4.7) applied once more on the final graph (r 3.5 µm, minlen 3, up to 5
  iterations), without the join, because tbext, relinefit and the P17 steps run after the first pass.
- **parallel duplicates.** Two linear tracks that stay within 3.5 µm of each other for ≥ 3 consecutive frames. The
  overlapping part of the shorter (or newer) one is removed, only if the overlap reaches its head or tail. Sister pairs are
  left alone.
- **FOV-border stubs.** Fork-free components of fewer than 6 nodes whose nodes all lie within 2 voxels of the lateral image
  border.

`shortbranch`, which deletes fork branches that end within 2 frames, was in P17 and is **off** in P20 and later. Its gain came
from two divisions, and it sits next to a cliff (maxk = 3 gives −0.0014).

| Evidence | Value |
|---|---|
| Local, 199 movies (P15 → P20) | +0.00028, CI [+0.00013, +0.00042]; divisions unchanged; 9/9 parameter neighbours positive; placebo p = 0.002 |
| Public / private | 0.000 / 0.000 |

### 4.10 Global-jump compensated relinking (`p22_jump.py`, `p_stage13.py`; P22A / P22B only)

Some acquisitions shift the whole field between two frames by 6–20 µm. At such transitions the false-negative rate of my links
rises to 4.7–9.5 %, against about 2 % normally.

**Shift estimate.** For each transition, the shift `v` is the mode of all node displacement vectors within 25 µm:

- a 3D histogram with 1 µm bins and 3×3×3 smoothing;
- refined by the mean of the vectors within 1.5 µm of the mode;
- at least 20 supporting pairs are required.

**Relinking.** Where |v| ≥ 8 µm, nodes are paired by Hungarian matching on |p_t + v − p_{t+1}| ≤ 3.5 µm. Single-child links
are replaced by the matched pairs; forks are never created or broken.

- Local +0.00010; 7/7 parameter neighbours positive; 5/5 random-direction placebos negative.
- Public and private: no gain. The unrounded public ordering put P20 ≥ P22A and P21 ≥ P22B, and the private scores are equal.

## 5. Configuration history

Every variant pushed to the Kaggle dataset is a JSON file in [`pstage/`](../pstage); P12's config was never pushed and is
not included. Some variants were evaluated or pushed to Kaggle but never submitted:

- P8: P6 with pruning moved after linking, the no-DSR twin of P7;
- P9, P10: junk-track pruning;
- P12: P10 with a 10 %-per-movie deletion cap, never pushed;
- P19 family: clean-up and DivNet candidates; P19R became P20;
- P23A, P23B: P23C / P23D + global-jump relinking.

The stage scripts used by pushed notebooks are kept, so each submission can be rerun with the exact script it used.
`p_stage5.py` (a P8 candidate-dump variant used for research) and `p_stage7.py` (P12 only) were internal drafts and are not
included.

| Script | Used by |
|---|---|
| `p_stage.py` | P1, P2 |
| `p_stage2.py` | P3, P4 |
| `p_stage3.py` | P5, P6 |
| `p_stage4.py` | P7, P8, P11, P13 |
| `p_stage6.py` | P9, P10 (`junk_prune`) |
| `p_stage8.py` | P14 |
| `p_stage9.py` | P15 |
| `p_stage10.py` / `p_stage11.py` | P16, P17, P18 |
| `p_stage12.py` | P19 family, P20, P21, P23C, P23D |
| `p_stage13.py` | P22A, P22B (and the P23A / P23B previews) |

Later scripts are supersets of earlier ones, with one exception: `junk_prune` is only implemented in `p_stage6.py`. A stage
script silently ignores config keys it does not implement. `notebooks/build_variant.py` checks for this.

| Config | Key settings |
|---|---|
| `p1_config.json` | th_start 0.90, th_stolen 0.97, prune before linking |
| `p2_config.json` | start-type only |
| `p3_config.json` | P1 + DSR 0.98 + dfork K=2 |
| `p4_config.json` | P1 + dfork K=2 |
| `p5_config.json` / `p6_config.json` | P3 / P4 + relink 0.65 + edge_link 0.4 (gap-2 on) |
| `p7_config.json` | P5 with post_prune after linking |
| `p8_config.json` | P6 with post_prune after linking (not submitted) |
| `p9_config.json` / `p10_config.json` | P8 + junk_prune / P9 + dfork K=100 (`p_stage6.py`; not submitted) |
| `p11_config.json` / `p13_config.json` | P8 + dfork K=100 / K=35 |
| `p14_config.json` | + term_trim + long_link |
| `p15_config.json` | + tbext + relinefit |
| `p15s85_config.json` | P15 with th_start 0.85 (the P21 threshold on the P15 base; not submitted) |
| `p17_config.json` | + cutdup, forkfrag, shortbranch |
| `p19s_config.json` / `p19r_config.json` | P17 + P19 clean-up / the same without shortbranch (= P20; not submitted) |
| `p20_config.json` | P15 + cutdup, forkfrag, dup heads, term_trim re-run (no join), parallel dups, FOV-border stubs |
| `p21s85_config.json` | **P21**: P20 with th_start 0.85 |
| `p21s80_config.json` / `p21s875_config.json` | P20 with th_start 0.80 / 0.875 (threshold scan; not submitted) |
| `p22a_config.json` / `p22b_config.json` | P20 / P21 + p22jump (run with `p_stage13.py`) |
| `p23a_config.json` / `p23b_config.json` | P22B with th_start 0.80 / 0.825 (run with `p_stage13.py`; not submitted) |
| `p23c_config.json` / `p23d_config.json` | P21 with th_start 0.80 / 0.825 |
| `p16a/p16b/p18/p19a/p19b/p19ra/p19rb_config.json` | DivNet variants. The DivNet weights are in the Kaggle P-stage dataset, not in this repository. Without them these configs abort the whole P-stage run at worker start, and the notebook then submits the B5 base. |
