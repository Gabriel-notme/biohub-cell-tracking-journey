# DivNet v2 (divnet2) -- cross-embryo fork scorer, LOEO end-to-end validation
Working dir /workspace/cl/p16/divnet2 (old pod). v1 files in ../divnet are read/imported only, never modified.

## PRE-REGISTRATION (2026-09-28 05:55 UTC, written before any v2 data was built, trained or scored)
Common to all v2 variants: same DivNet architecture as v1 (divnet_lib.DivNet width 32, 5 frames t-2..t+2, crop (16,48,48) on
the (1,2,2) grid, p / a+b marker channels), same augmentation, same batch (128, 1/3 positive slots), final checkpoint (no
selection), 3 seeds {0,1,2} (= v1's ensemble size, so the comparison isolates the data/recipe change and the scoring cost is
unchanged). LOEO: a model for test embryo T is trained ONLY on rows of the other embryo E (v1 'full' rows of E + new rows of E).
New rows (build_v2rows.py -> /dev/shm/divnet2/<movie>.npz, same crop function divnet_lib.crops):
  PS  pseudo-positives: every P15-final-graph fork of the movie labelled 'Fnc' in nm/dv_cand (not covered by GT), one row per fork
      (Ftp / Ffp forks are already in v1's pool rows). Soft target 0.8 (~ evaluable P15 fork precision 0.62-0.73).
  PR  pseudo-rivals: for each PS fork (p,a,b) and each daughter d: the 2 nearest P15-graph nodes r at t+1 within 16 um of p that
      are not children of p and have a parent -> (p, d, r), target 0 (mirrors v1's GT rival negatives).
  HN  hard negatives: from the real first-pass division-completion pool of the movie (dnlog of v1 run dnA; candidate set and
      b1dep score 'fork_b1' do not depend on DivNet), labels joined from nm/dv_cand: candidates labelled N or X, top-25 per type
      (start / stolen) by b1dep, target 0. (Caveat: b1dep saw both embryos; it is used only to SELECT which labelled negatives of
      the training embryo are included; labels are the training embryo's own.)
Variants (3; no others will be run):
  V2H : v1 full rows + HN; 2000 steps, lr 1e-3 cosine (v1 schedule); 25% of negative slots drawn from HN.
  V2P : stage 1 (pretrain) 2000 steps lr 1e-3: positive slots 50% PS (target 0.8) / 50% real positives (v1 y=1), negatives = v1
        negatives + PR (uniform); stage 2 (fine-tune) 1000 steps, peak lr 3e-4 cosine, v1 full rows only.
  V2PH: V2P with HN added in both stages (25% of negative slots).
  Not run: longer context t-3..t+3 (full crop rebuild), 5-seed ensembles (scoring time).
End-to-end evaluation: v1's dnA form exactly: dn_driver (copied here as dn2_driver.sh, only paths changed) with DN_MODE=divnet,
DN_SHORT=b1loeo:300 (shortlist = top-300/type by the LOEO b1 recipe), DN_MAP=rank (accepted count = P15's), P15 config, all 5 sets
(199 movies), DN_MODELS = the variant's 3 LOEO checkpoints per embryo. Scored with the official metric (evalx.score_movie) into
./rev. Reference v1 = existing dnA run (cl/rev/dn_dnA_*.json) AND an A/A rerun of v1 in the same form (tag v1rr) to measure
run-to-run noise.
Decision rule ("multiple checks"; a variant PASSES only if ALL hold):
  M1 all-199 delta(v2 - v1) > 0 and paired movie-bootstrap (2000 resamples) Bonferroni-adjusted CI (alpha 0.05/3 -> 98.33%)
     lower bound > 0.
  M2 each embryo delta >= 0.
  M3 house bar strict.py S1-S7 all ok on the v2-vs-v1 rows (S1 95% CI>0, S2 per-embryo P>0>=.95, S3 clean40>0, S4 B5clean72>=0,
     S5 leave-one-set-out all >0, S6 drop top-5 / per-embryo top-3 contributors, S7 movies up > down).
  M4 robustness to run noise: M1 also holds against the v1 A/A rerun (v1rr), and |delta(v1rr - v1)| < delta(v2 - v1).
  M5 candidate level on the real dc pool (first pass, dnlog labels via dv_cand): per embryo, v2's global top-k (k = P15 acceptance
     count per type) net TP - 0.4 FP summed over types >= v1's.
If several variants pass, take the largest all-199 delta. If none passes: verdict NO (no deploy training).
Deploy only on PASS: same recipe on all 199 movies (rows of both embryos), 3 seeds, fixed steps, final checkpoint ->
./deploy, drop-in divnet_core_v2.py (DivNetScorer interface of /workspace/p17ds/divnet_core.py), timing for 200 candidates.

## LOG
- 05:59 UTC: built new rows (build_v2rows.py): PS 3147, PR 11916, HN 7176 over 199 movies. Launched A/A rerun of v1 (tag v1rr) in dnA form.
- 06:01 UTC: smoke test OK; launched LOEO training of V2H/V2P/V2PH x {44b6,6bba} x seeds 0,1,2 (train_<V>_tr<E>.log).
- 06:04 UTC: V2H_tr44b6 OOM'd on GPU0 (shared with the v1rr e2e run); relaunched via waitlaunch.sh (starts when a GPU has 24 GB free). Same recipe, nothing else changed.
- 06:12 UTC: A/A so far: v1rr hold36 / audit32 submission SHA identical to the original dnA runs (24c3ded6..., 8c42fdf6...) ->
  the e2e pipeline is deterministic; run-to-run noise ~0 (M4 reduces to M1).
- v1 3-seed ensemble, candidate level on P15 pool rows (cand_metrics.py, LOEO, all): 44b6 start AUC .786 stolen .714, GT AUC .862;
  6bba start .651 stolen .604, GT .761 (prec@P = 0 for stolen in both).
- 06:17 UTC: A/A rerun v1rr finished: all 5 set scores identical to dnA (hold36 .983454, prev4 .949432, audit32 .971691, t127a .961804, t127b .946837); e2e is deterministic, A/A delta = 0.
- 06:39 UTC: V2PH_tr44b6 OOM'd after saving seed 1 (another process briefly took GPU memory); seed 2 relaunched alone via waitlaunch.sh (same seed => same recipe).
- 06:45 UTC: V2H training finished (6 ckpts); launched V2H e2e (dnA form, GPU1).
- 06:50 UTC: V2P training finished; launched V2P e2e (GPU0).
- 06:53 UTC: V2PH training finished; launched V2PH e2e (GPU1).
- 06:56 UTC secondary, candidate level on P15 pool rows (cand_metrics.py, 3-seed LOEO ensembles, weighted AUC; full table cand_metrics_all.txt):
            44b6(tr 6bba): start  stolen  GT   | 6bba(tr 44b6): start  stolen  GT
    v1ens                  .786   .714   .862 |                .651   .604   .761
    V2H                    .978   .670   .856 |                .557   .642   .754
    V2P                    .810   .498   .833 |                .837   .623   .865
    V2PH                   .939   .519   .864 |                .786   .575   .875
  Pseudo-labels strongly help the positives-starved direction (44b6-trained on 6bba: GT AUC .761 -> .865/.875, start .651 -> .84/.79)
  but hurt stolen-type ranking on 44b6 (.714 -> .50/.52) and push already-recovered duplicates (D) up (Dhi .62 -> .85/.73).
  prec@P for stolen stays 0 for every scorer. (Pool rows use random N/X negatives, not the b1loeo top-300 shortlist: e2e decides.)
- 07:05 UTC RESULT V2H (judge_V2H.txt): vs v1 all-199 -0.00049, Bonferroni CI [-0.00487,+0.00375], div TP +0 FP +2;
  44b6 -0.00233, 6bba -0.00010; clean40 -0.00540; strict FAIL S1-S7 (up 28 / down 44); M5 ok (44b6 net 1.8 -> 3.0, 6bba -9.2 -> -9.0).
  -> V2H FAILS (M1, M2, M3, M4).
- 07:09 UTC RESULT V2P (judge_V2P.txt): vs v1 all-199 +0.00136, Bonferroni CI [-0.00348,+0.00639] (P>0 .74), div TP +2 FP -2;
  44b6 +0.00094, 6bba +0.00145; per set hold36 -0.00592, prev4 -0.00542, audit32 +0.00006, t127a +0.00439, t127b +0.00469;
  clean40 -0.00603; strict FAIL S1-S6 (S7 ok, up 67 / down 26); M5 FAIL (6bba net -9.2 -> -9.6). -> V2P FAILS (M1, M3, M4, M5).
- 07:11 UTC RESULT V2PH (judge_V2PH.txt): vs v1 all-199 +0.00010, Bonferroni CI [-0.00418,+0.00461] (P>0 .51), div TP +0 FP -1;
  44b6 +0.00085, 6bba -0.00006; per set hold36 -0.00652, prev4 0, audit32 -0.00004, t127a +0.00206, t127b +0.00288; clean40 -0.00595;
  strict FAIL S1-S6 (S7 ok); M5 ok (6bba first-pass top-k FP 38 -> 22 at TP 6 -> 6; 44b6 net 1.8 -> 3.6). -> V2PH FAILS (M1-M4).

# VERDICT (pre-registered rule): NO. None of V2H / V2P / V2PH beats v1 end-to-end under LOEO. No deploy training was run
# (deploy/ is empty by design). divnet_core_v2.py / time_scorer.py were prepared but are not needed.
Summary (v2 - v1, dnA form, all 199, official metric; Bonferroni 98.33% CI; clean40 = hold36+prev4):
  V2H  -0.00049 [-0.00487,+0.00375]  clean40 -0.00540  div TP +0 FP +2   44b6 -0.00233 6bba -0.00010
  V2P  +0.00136 [-0.00348,+0.00639]  clean40 -0.00603  div TP +2 FP -2   44b6 +0.00094 6bba +0.00145
  V2PH +0.00010 [-0.00418,+0.00461]  clean40 -0.00595  div TP +0 FP -1   44b6 +0.00085 6bba -0.00006
  A/A (v1 rerun) = 0.00000 exactly (deterministic pipeline).
Findings:
 1. Pseudo-label pretraining fixes the candidate-level weakness of the 44b6-trained direction (GT-triple AUC on 6bba .761 -> .87,
    first-pass top-k FP on 6bba 38 -> 22 for V2PH), but this does not reach the end-to-end score: in dnA form the division stage only
    changes a handful of accepted forks per 199 movies (|div dTP|, |dFP| <= 2 for every variant), and the edge side of those changes
    dominates the score deltas.
 2. The LOEO e2e comparison in dnA form is underpowered: the paired movie-bootstrap 95% CI half-width is ~0.0035 for any variant,
    i.e. only an improvement of roughly 6+ net division TPs over 199 movies could pass; v1's dnA itself is only +0.00009 vs dc disabled.
 3. Every variant loses ~0.006 on hold36 (2 TP fewer than v1 there) while gaining on t127a/b: set-level swings of +-0.006 appear
    without any consistent direction, which is what the Bonferroni/strict checks are there to catch.
- Integrity: all 15 v2 set runs report "errors": 0 / "p15_errors": 0, every worker log shows DN_INSTALLED with the variant's own
  model glob, no Traceback/OOM in worker logs; dnlog files per set match dnA's.

## Stricter multi-check tool for P-stage candidates (user request: more rigorous, multiple checks, also for P19)
multicheck.py <rows.json> [m] [label]: G1 bootstrap CI > 0 at Bonferroni alpha 0.05/m (4000 resamples) + CI lower bounds for
m = 1,3,5,10,20; G2 each embryo > 0 with P>0 >= .95; G3 strict.py S1-S7; G4 split-half replication (both random embryo-stratified
halves > 0 in >= 80% of 2000 splits); G5 one-sided sign test on movies with evaluable changes, p < 0.05/m. Applied at 07:18 UTC:
  P17 vs P15 (rows_p17_full.json): +0.00060, CI m=1 [+0.00013,+0.00134], lower bound still > 0 at m=20 (+0.00004); 44b6 +0.00152
    (P 1.000), 6bba +0.00040 (P .991); split-half 1.000; sign test 14 up / 3 down p .0064 -> PASS all gates (mc_p17_vs_p15.txt).
    Caveat unchanged: this is local (2-embryo) evidence; the effect is small (+0.0006) and the public LB rounds to 0.001.
  P18 vs P17 (the DivNet-blend step alone; rows_p18_vs_p17.json via mk_rows): +0.00471 but FAIL: S2 (44b6 P .92), S6, sign test
    17 up / 12 down p .23, CI lower bound < 0 already at m=3 -- and this is IN-SAMPLE (dn1_full_trall weights were trained on
    all 199 movies), while the only cross-embryo test of that form (v1 dnC, LOEO) was -0.00537 vs P15 and today's LOEO v2
    results show no transferable gain. P18's local +0.0053 vs P15 is therefore mostly an in-sample DivNet effect (mc_p18.txt).
