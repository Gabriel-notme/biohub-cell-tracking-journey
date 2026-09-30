# divnet (P16) notes
Goal: cross-embryo (LOEO) fork scorer vs b1 inside P15 div_complete; + fork veto test.

## Data
- b1-recipe data: artifact prepare_events.prepare_one on all 199 movies -> /dev/shm/divnet_ev (7.8G); LOEO dirs /dev/shm/divnet_ev_tr{44b6,6bba}
  (split.json: train = embryo minus 6, calibration = 6 random movies of same embryo). prep_b1data.py
- DivNet rows: build_divnet.py -> /dev/shm/divnet/<movie>.npz (31G). GT rows (all GT divisions + 2 rival negatives/parent, 200/movie)
  + P15 pool rows from nm/dv_cand (P=1, N/X=0 subsampled N 250, X 150 per movie with inverse-rate weight, D=-1 eval only, forks Ftp/Ffp).
  Counts: 44b6 GT div 26 (!), pool P start 4 stolen 16, Ftp 13 Ffp 8; 6bba GT div 125, P start 15 stolen 56, Ftp 54 Ffp 20.
  => positives are very scarce (44b6-trained model sees 59 positives, 6bba 250).

## Baseline b1 recipe retrained LOEO (train_events.py unchanged, 8500 steps, ~7.5 min each on GPU0)
Candidate-level on the P15 remaining pool (cand_metrics.py; AUC weighted, N/X inverse-rate):
- deployed b1 (in-sample on t127/audit32): all: 44b6 start AUC .988 stolen .712 veto .500 | 6bba start .989 stolen .814 veto .762
  clean40: 44b6 stolen .549 veto .528 | 6bba stolen .743 veto .750
- b1-LOEO last.pt: all: 44b6 start .932 stolen .548 veto .519 | 6bba start .787 stolen .636 veto .630
  clean40: 44b6 stolen .494 | 6bba stolen .449  -> the recipe transfers poorly across embryos (b1dep's pool AUC is in-sample-inflated).

## Candidate level, DivNet dn1 seed0 (width 32, 2000 steps, full arm), LOEO, P15 remaining pool
- 44b6 (model trained on 6bba): start AUC .771, stolen .671, veto .490, GT-triple AUC .892
- 6bba (model trained on 44b6, only 59 positives): start .598, stolen .587, veto .742, GT .768
- prec@P for stolen = 0 for every LOEO scorer (b1-LOEO too); in-sample DivNet = 1.000 (memorised).

## End-to-end: b1 recipe retrained LOEO inside P15 div_complete (dn_driver.sh b1lr, DN_MAP=rank: per movie/type the i-th best
LOEO candidate gets b1dep's i-th best prob, so th 0.9/0.97 accept the same count as P15; only the choice changes)
- all -0.01240 CI [-0.018,-0.007]; div TP -15 FP +25; clean40 -0.01109; 44b6 -0.01609, 6bba -0.01155; strict FAIL all.
- => deployed b1 (trained on both embryos, 159 movies) is FAR better in-embryo than the same recipe trained cross-embryo, even on
  clean40 where b1dep is out-of-sample but in-embryo. Division completion's local value is largely an in-embryo effect.
- Actual dc pool (dnlog, analyze_dnlog.py): b1dep global top-k TP/FP 44b6 start 3/0 stolen 4/2, 6bba start 6/6 stolen 7/7;
  b1-LOEO top-k 44b6 1/2, 1/4; 6bba 0/15, 0/7.

## Reference: P15 with division completion disabled (cfg/p15nodc_config.json: th_start=th_stolen=1.01), tag nodc
- P15 vs nodc: dc is worth +0.00838 locally (div TP +17 FP +7); clean40 +0.0079 (hold36) / +0.0013 (prev4).
- b1lr (LOEO b1 recipe, P15 acceptance counts) vs nodc: -0.00402 (div TP +2, FP +32); 44b6 +0.00137, 6bba -0.00530, hold36 -0.0012.
  => with a CROSS-EMBRYO fork scorer, division completion as configured in P15 adds ~0 TP and many FP. On new embryos (hidden test)
  P15's dc stage may be neutral or negative (b1dep is trained on 2 embryos/159 movies, so reality is between the two numbers).
  Consistent with local P6->P15 gains not showing on the public LB.

## DivNet deployability problem found
- The real div_complete pool has ~1e5 candidates per movie (mostly 'U', unannotated); a per-candidate 3D crop is infeasible
  (first dnr run killed). DivNet is therefore run on a shortlist: top-300 per type by b1-LOEO (dnA, LOEO-pure) or by b1dep
  (dnB, deployable form; b1dep in-sample on t127/audit32) and re-ranks it; rank-mapped counts as in b1lr.
- GPU1 is full (other agent) -> first dnA/dnB attempt OOM'd on GPU1 workers; relaunched all workers on GPU0 (DN_GPU=0).

## Fork veto (dn_veto.py via rule_eval on P15 graphs; LOEO DivNet 3-seed ensemble; pre-registered th 0.02 / 0.05; cut the
child with the shorter forward chain)
- th .02: all -0.00520 CI [-0.0099,-0.0008] div TP -13 FP -12; clean40 -0.00278; FAIL S1-S7.  th .05: all -0.00541; FAIL.
- The 44b6-trained model (59 positives) is badly calibrated on 6bba forks (median 0.064; 36% of forks < 0.02).

## b1 recipe LOEO with P15's RAW thresholds (b1lraw; what P15's dc does when its fork model has never seen the embryo)
- vs P15: -0.02145 (div TP -17 FP +99); vs nodc: -0.01307 (div TP +0 FP +106) -> the LOEO b1 is over-confident on the other
  embryo (many candidates >= 0.9/0.97). (driver bash got corrupted by an edit mid-run; hold36 re-scored manually, errors 0.)

## dnA / dnB end-to-end (partial, sets finished so far; per set score, div TP/FP)
            P15(b1dep)        nodc              b1lr (b1-LOEO)     dnA (DivNet-LOEO rerank of b1-LOEO top300)
  hold36    0.98375 16/9      0.97584 12/7      0.97466 12/9       0.98345 15/7
  audit32   0.98817 13/2      0.96661  8/2      0.96113  9/9       0.97169 10/4   (b1dep in-sample on audit32)
  prev4     0.95072  1/1      0.94943  1/1      0.92356  0/2       0.94943  1/1
- Recipe-level LOEO comparison: DivNet (dnA) >> b1 recipe (b1lr) cross-embryo: hold36 +0.0088, audit32 +0.0106, prev4 +0.0259;
  on clean hold36 the fully-LOEO dnA nearly matches in-embryo b1dep (-0.0003).

## PRE-REGISTERED (01:37 UTC, before any dnC result): dnC = b1dep top-300 shortlist per type, score = sigmoid(0.5 logit(b1dep) +
0.5 logit(DivNet-LOEO 3-seed mean)), rank-mapped to b1dep's counts (P15 th 0.9/0.97). Motivation: deployable form keeps b1dep's
ranking information and adds the cross-embryo DivNet opinion. Weight 0.5 fixed a priori; no other variants will be tried.

## dnA final (DivNet-LOEO re-rank of b1-LOEO top-300; fully LOEO) -- rows_dnA*.json
- vs P15: all -0.00829 CI [-0.0137,-0.0035], div TP -12 FP +9; hold36 -0.00029, prev4 -0.00129, audit32 -0.0165, t127a -0.0094,
  t127b -0.0103. strict FAIL S1-S7 (P15's b1dep is in-sample on t127/audit32 and in-embryo everywhere).
- vs b1lr (recipe vs recipe, both LOEO): all +0.00411 CI [+0.0004,+0.0082], div TP +3 FP -16; clean40 +0.01065; 44b6 +0.0071
  (P 0.997), 6bba +0.0034 (P 0.930) -> strict fails only S2 (6bba P<0.95). DivNet recipe transfers better than the b1 recipe.
- vs nodc: all +0.00009 (44b6 +0.0085 with the 6bba-trained model [250 positives]; 6bba -0.0019 with the 44b6-trained model
  [59 positives]); t127a -0.0051, t127b -0.0020, hold36 +0.0076, audit32 +0.0051.

## dnB final (DivNet-LOEO re-rank of b1dep top-300, rank-mapped) -- rows_dnB.json
- vs P15: all -0.00746 CI [-0.0127,-0.0030], div TP -10 FP +10; hold36 +0.00095, prev4 0.00000, audit32 -0.0126, t127a -0.0083,
  t127b -0.0116; clean40 +0.00085 (44b6 +0.00175, 6bba +0.00012). strict FAIL (S1,S2,S4,S5,S6,S7; only S3 ok).
- vs nodc: +0.00092.
- Actual dc pool (dnlog), global top-k by type: DivNet-LOEO on 44b6 (trained on 6bba, 250 pos) ~ b1dep; on 6bba (trained on 44b6,
  59 pos) stolen top-345 TP 1 FP 20 vs b1dep 8/7 -> the weak direction is the 44b6-trained model (too few positives).

## dnC (pre-registered logit-mean b1dep+DivNet) partial: hold36 0.98219 (15/8) vs P15 0.98375; audit32 0.97781; prev4 = P15.

## Runtime (profile, RTX PRO 6000 under load, one movie with 382k candidates): candidates() 1.6 s (already in P15), load all
frames 1.5 s, 600 crops 0.3 s, 3-model DivNet on 600 crops 2.1 s -> ~4 s/movie extra here; on a T4 likely 5-10 s (over the 3 s
budget unless K or the number of seeds is reduced). Per-candidate crops are the cost driver (b1 uses shared per-node embeddings).

## Deploy weights trained anyway (models/dn1_full_trall_s{0,1,2}.pt, all 199 movies, same recipe) for a possible follow-up;
divnet_draft_NOT_ACCEPTED.py = self-contained scorer + dnC-form install(), NOT validated/accepted (strict FAIL).

## dnC final (pre-registered logit-mean) -- rows_dnC.json
- vs P15: all -0.00537 CI [-0.0095,-0.0017], div TP -7 FP +8; clean40 -0.00145 (44b6 -0.00517, 6bba +0.00130); FAIL S1-S7.
- vs nodc: +0.00300.

# SUMMARY (end-to-end, official metric, delta vs P15 0.97000 on all 199; clean40 = hold36+prev4)
  variant                                   all       clean40   div dTP/dFP   strict
  nodc (dc disabled)                       -0.00838   -0.00739  -17/-7        FAIL all
  b1lr  (b1 recipe LOEO, rank-mapped)      -0.01240   -0.01109  -15/+25       FAIL all
  b1lraw (b1 recipe LOEO, raw th .9/.97)   -0.02145   -0.01659  -17/+99       FAIL all
  dnA (DivNet LOEO rerank b1-LOEO top300)  -0.00829   -0.00044  -12/+9        FAIL all (vs b1lr: +0.00411, fails only S2)
  dnB (DivNet LOEO rerank b1dep top300)    -0.00746   +0.00085  -10/+10       FAIL (only S3 ok)
  dnC (dnB + logit mean with b1dep)        -0.00537   -0.00145  -7/+8         FAIL all
  veto th .02 / .05 (P15 forks)            -0.00520 / -0.00541              FAIL all
VERDICT: FAIL. No variant beats P15 under LOEO. Losses on t127/audit32 are partly b1dep in-sample leakage in P15, but even on
clean40 the best (dnB) is +0.00085 (1 FP fewer), far from significant.
Findings worth keeping:
 1. The fork scorer is strongly embryo-specific: the b1 recipe trained on one embryo turns P15's division completion from +0.0084
    (vs nodc, in-embryo b1dep) into -0.004 (same counts) / -0.013 (raw thresholds, +106 FP). P15's dc gain on NEW embryos is
    therefore uncertain and could be ~0 or negative; a public-LB probe of P15 with dc disabled (cfg/p15nodc_config.json, local
    -0.0084) would measure it directly.
 2. DivNet (event-centred 3D crop + markers) transfers better than the b1 recipe (dnA vs b1lr +0.0041, CI>0; clean40 +0.0107),
    and the 6bba-trained direction (250 positives) is good on 44b6 (dnA vs nodc 44b6 +0.0085). The 44b6-trained direction (59
    positives) is poor. Positives are the bottleneck (26 GT divisions in 44b6, 125 in 6bba).
 3. Per-candidate crops cost ~4 s/movie for 600 shortlisted candidates (1e5 candidates per movie exist) -> needs a shortlist.
