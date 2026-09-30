# b1ens (P16): is the b1 fork scorer's cross-embryo quality limited by seed variance and/or training-set size?
Started 2026-09-28 02:08 UTC. Reuses divnet tools/data (/dev/shm/divnet_ev*, dn_patch.py logic, analyze_dnlog.py labels, dn_driver.sh flow).

## Setup (fixed before any training)
- Recipe: /workspace/models/b34/train_events.py unchanged, args identical to divnet b1loeo_tr* (context 3, width 24, batch 128,
  lr 2e-4, 8500 steps, schedule steps, workers 6), except --seconds 100000 (divnet used 2400 but finished in 450 s, so its LR
  schedule was purely step-based; with ~16 concurrent runs the wall-clock term must not kick in). last.pt, no selection.
- Seeds: s0 = 20260921 (= existing divnet b1loeo_tr{E}/last.pt, symlinked as models/loeo_tr{E}_s0), s1..s4 = 20260922..20260925.
- LOEO data: /dev/shm/divnet_ev_tr{E} (train = embryo minus 6 calibration movies: 44b6 65, 6bba 122).
- Half arm (data scaling): /dev/shm/b1ens_half{k}, k=0,1,2: train = sorted(random.Random(20260928+k).sample(6bba LOEO train, 61));
  same 6 calibration movies; seed s_k (paired with full-arm s_k). Same 8500 steps (fixed compute).
- Deploy arm (SPECULATIVE, trained now only to save wall-clock; used ONLY if the decision in Q3 passes): /dev/shm/b1ens_all:
  train = all 199 movies, calibration = the 12 LOEO calibration movies (in-sample, monitoring only), seeds s0..s4, last.pt.
- End-to-end: P15 P-stage (p15_config.json unchanged) on the original B5 lineage graphs, 5 sets (hold36, prev4, audit32, t127a,
  t127b = 199 movies). div_complete fork scores replaced by the LOEO scorer (44b6_* movies scored by models trained on 6bba only
  and vice versa), DN_MAP=rank (per movie and type, the i-th best candidate by the LOEO scorer receives b1dep's i-th best prob,
  so P15's th .9/.97 accept the same number of forks; only the choice changes). Ensemble = MEAN FORK LOGIT over the checkpoints
  (EventRefiner 'mean'; P15's event_config fork_ensemble 'first' is overridden to 'mean' for the LOEO refiner only).
  Scoring: review.py scoreg (official metric) -> mk_rows.py -> strict.py. Code: b1e_patch.py (copy of dn_patch.py, multi-ckpt),
  b1e_driver.sh (copy of dn_driver.sh flow, more workers per set).
- Candidate level: an_dnlog.py (copy of analyze_dnlog.py) on the pass-1 div_complete pool (identical across variants), labels
  from nm/dv_cand; per test embryo and type: global top-k TP/FP with k = b1dep acceptance count (th .9/.97), AUC.

## PRE-REGISTERED (02:13 UTC, before any new model is trained or any new result is seen)
Q1 seed ensemble. Variants S0..S4 (single LOEO seed), E5 (mean logit of the 5 seeds). S0 is rerun through b1e_driver (must equal
divnet's dn_b1lr per movie; if not, report it and use the rerun as S0).
  PASS Q1 iff all of:
   (a) E5 vs S0: all-199 delta > 0 and paired movie-bootstrap 95% CI lower bound > 0 (strict.py S1);
   (b) E5 vs S0 per embryo (= per training direction) delta >= 0 in both embryos;
   (c) E5 > mean of the 5 single-seed scores on all 199, and E5 >= that mean in each embryo (guards against an unlucky S0).
  Always reported: all strict.py lines for E5 vs S0, min/mean/max of S0..S4 per embryo, E5's rank among singles, candidate-level
  top-k TP/FP per embryo/type for S0..S4 and E5.
Q2 data scaling (6bba-trained -> tested on the 71 44b6 movies). full = S0,S1,S2 restricted to 44b6 movies; half = H0,H1,H2
(44b6-only P-stage runs, same code, 6bba movies not run).
  PASS Q2 iff all of: mean over the 3 seed pairs of score(full_k) - score(half_k) on 44b6 movies > 0; >= 2 of 3 paired deltas > 0;
  movie bootstrap (resample 44b6 movies, mean paired delta over seeds) P(>0) >= 0.90; and candidate level does not contradict
  (mean over seeds of top-k (TP - FP) on the 44b6 pool, both types summed: full >= half).
Q3 decision: train/keep the deploy ensemble ONLY if Q1 AND Q2 pass. Otherwise decision NO (speculative deploy weights stay
unused, labelled NOT ACCEPTED). If yes: deploy form = rank-mapped (same acceptance count as b1dep at th .9/.97, choice by the mean
logit of the 5 all-199 seeds + deployed b1 = 6 models, equal weight); sanity rerun on 199 movies is IN-SAMPLE and reported as such.
No thresholds or weights are tuned on test embryos; no other variants will be tried.

## Log
- 02:14 UTC launched 16 trainings (train_all.sh): loeo_tr{44b6,6bba}_s1..4, half6bba_s0..2, deploy b1all_s0..4 (speculative).
- 02:17 UTC launched S0 rerun (b1e_driver.sh s0, existing divnet seed) while trainings run.
- 02:27 UTC S0 rerun via b1e_driver: submission SHA identical to divnet dn_b1lr on all 5 sets (pipeline deterministic; S0 == b1lr).
- 02:28 UTC [20-min report] 16 trainings at step ~3900/8500 (ETA ~02:45, GPU-bound, 8 per GPU); S0 end-to-end done; analysis scripts q1.py q2.py cand.py time_dc.py ready.
- 02:39 UTC first eval launch (132 P-stage workers) hit CUDA OOM (4.4 GB/worker; deploy trainings still on GPU) -> movies fell back with errors; killed at 02:44, outputs moved to trash_oom/ (never scored). Added torch.cuda.empty_cache() at the end of the patched scorer (no numeric effect) + expandable_segments, relaunched 02:45 with 80 workers.
- 02:48 UTC [20-min report] all 16 trainings done (TRAIN_DONE, 8500 steps, ~25 min each concurrently); 8 end-to-end runs (S1-S4, E5 full; H0-H2 44b6-only) relaunched 02:45 after OOM, ETA ~03:05.
- 03:02 UTC NOTE: container CPU quota is 27.2 cores (/sys/fs/cgroup/cpu.max 2720000/100000) although nproc=128; P-stage runs are CPU-quota bound (per-movie 2.7x slower than S0 with 80 workers). ETA of the 8 runs ~03:25-03:35.
- 03:08 UTC [20-min report] 8 end-to-end runs ~45% by movie count (CPU-quota bound), no worker errors; no Q1/Q2 result seen yet.
- 03:17 UTC all 8 runs complete: errors 0 in every set; E5 workers loaded 5 ckpts per embryo, singles 1 (B1E_LOADED).

## RESULTS (official metric, end-to-end P15 P-stage, rank-mapped; seen 03:18 UTC)
### Q1 seed ensemble (LOEO; 44b6 movies scored by 6bba-trained models, 6bba movies by 44b6-trained models)
              all(199)          44b6(71)          6bba(128)         clean40           div TP/FP (all)
  S0          0.95760           0.95260           0.95865           0.96958           52/53
  S1          0.95689           0.95018           0.95834           0.96954           51/55
  S2          0.95619           0.95204           0.95709           0.96953           50/57
  S3          0.95831           0.95767           0.95836           0.97136           52/48
  S4          0.95775           0.95376           0.95860           0.97286           53/57
  singles     mean .95735 sd .00082 [.95619,.95831] | 44b6 mean .95325 sd .00279 [.95018,.95767] | 6bba mean .95821 sd .00064
  E5          0.95642 (rank 5/6) 0.95358 (3/6)    0.95696 (6/6)     0.96923 (6/6)     52/63
  ref P15     0.97000           0.96870           0.97020           0.98067           67/28   (b1dep: in-embryo, in-sample on t127/audit32)
  ref nodc    0.96162           0.95123           0.96395           0.97328           50/21
  E5 - S0: all -0.00118 CI [-0.00312, +0.00058]; 44b6 +0.00098 (P>0 .765); 6bba -0.00169 (P>0 .058); clean40 -0.00035;
           div TP +0 FP +10; strict FAIL S1-S7 (rows_E5_vs_S0.json).
  E5 - mean(singles): all -0.00093; 44b6 +0.00033; 6bba -0.00125; clean40 -0.00134.
  Pre-registered Q1: (a) FAIL (delta<0, CI includes 0), (b) FAIL (6bba -0.00169), (c) FAIL (E5 below single mean) -> Q1 FAIL.
  Candidate level (pass-1 dc pool, global top-k per embryo/type, k = b1dep acceptance count; cand_*.txt), sum(TP-FP) over 4 cells:
    S0 -28, S1 -36, S2 -20, S3 -28, S4 -31 (mean -28.6), E5 -21; b1dep +3 (in-embryo). AUC E5 vs mean single: 44b6 start .941/.931,
    44b6 stolen .643/.640, 6bba start .748/.736, 6bba stolen .680/.671. => the usual small ensemble gain at candidate level
    (fewer top-k FP), but every LOEO scorer is far below b1dep and it does not carry to the end-to-end score.
### Q2 data scaling (6bba-trained -> 44b6 movies, 71 movies; full = 122 train movies, half = 61, same 8500 steps)
  seed0 full .95260 (div 9/13) half .95060 (9/17) +0.00200 | seed1 .95018 (8/13) vs .95007 (8/13) +0.00011 |
  seed2 .95204 (9/15) vs .95270 (8/9) -0.00066 ; mean +0.00048, 2/3 positive, movie bootstrap P(>0) 0.590 CI [-0.0035, +0.0056]
  (clean40 44b6 +0.00122, t127+audit 44b6 -0.00054).
  Candidate level (44b6 pool, sum TP-FP start+stolen): full -4,-4,-2 (mean -3.3) vs half -6,-7,-2 (mean -5.0); stolen AUC full
  .647 vs half .616, start .942 vs .931 -> slight, non-decisive advantage for more data.
  Pre-registered Q2: mean>0 ok, 2/3 ok, bootstrap P .59 < .90 FAIL -> Q2 FAIL.
### Q3 DECISION: NO. Neither pre-registered condition holds, so the deploy ensemble is NOT accepted and no INTEGRATION.md is written.
  The 5 speculative all-199 weights (deploy/b1all_s{0..4}/last.pt, trained 02:14-02:42, 8500 steps each) are kept but labelled
  NOT ACCEPTED (deploy/NOT_ACCEPTED.txt). Loss curves (loss_curves.py): edge loss 0.11-0.12 -> 0.011-0.018, fork 0.13-0.16 ->
  0.018-0.039 (mean per 1000-step block, last block), smooth, no divergence; in-sample calibration fork AP 1.0.
  Informational runtime (time_dc.py, 3 largest t127a movies, RTX PRO 6000, CPU quota 27 cores): dc scoring P15 single b1 46.3/34.3/28.8 s
  vs 6-model ensemble (b1dep + 5 all-199) 59.9/46.1/38.7 s -> +10..14 s per large movie (+30%), more on a T4.
### Interpretation
  - Seed variance: single-seed spread (all-199 sd 0.0008, range 0.0021; 44b6 range 0.0075 because only 26 GT divisions) is small
    against the LOEO gap to P15 (-0.012) and to nodc (-0.004). Averaging 5 seeds does not lift the end-to-end score (E5 is 5th of
    6 on all-199) -> the cross-embryo weakness is not seed noise.
  - Training-set size within one embryo: doubling 61 -> 122 6bba movies gives +0.0005 on 44b6 (n.s., P .59) and a small candidate-
    level gain -> more same-embryo data barely helps cross-embryo. The limiting factor is the embryo shift (appearance/annotation
    differences between embryos), which neither more seeds nor more movies of the same embryo fix. Whether a 2-embryo training set
    (deployed b1, P1 local-vs-public parity) generalises to a 3rd embryo cannot be tested LOEO with 2 embryos.
  - No evidence that an all-199 seed ensemble would beat the deployed b1 on new embryos; keep P15's b1.
- Note: per-movie outputs do not depend on shard composition (S0 with 3/1/3/5/5 workers per set reproduced divnet's 2-worker b1lr SHA exactly), so 44b6-only half runs are comparable with the 44b6 movies of the full runs.
- 03:27 UTC [final report] Q1 FAIL, Q2 FAIL, decision NO. Pod idle (no GPU jobs).
