# linknet (P16) notes
Goal: appearance-based pair/association model to fix P15 residual link errors cross-embryo (LOEO).
- prep.py: 2x2-avgpooled uint8 volumes in /dev/shm/linknet/vol (T,64,128,128), per-movie node/pair tables in tab/.
  Pairs = all evaluable (v=1) pairs touched by rl_cands rows (s-d, s-c, q-d, q-c), label = official-style tp.
- Baselines on held-out pool (rl_b1an.py): b1 b_sd AUC(y) clean 44b6 .872 / 6bba .838; in-sample 44b6 .829 / 6bba .786.
  LightGBM geometry LOEO (nosw log): 6bba .855, 44b6 .900 ; precision at top essentially zero (nothing reaches p>=.5).
- Held-out LOEO baselines on the same rows (geo_lgb.py, LightGBM 500 rounds, target y): AUC geo 44b6 clean .943 / insample .926,
  6bba clean .928 / insample .902; geo+b1 about the same (+.005 clean). Precision at p>=.3: ~4/23 clean, 26-29/95 insample (<30%).
- Pool positives (y=1) are 70% "flip" (rl_oracle: removed child/parent within 7um of the GT node = matching ambiguity):
  44b6 clean 83 flip/38 real, ins 336/69; 6bba clean 159/74, ins 879/467.
- v1 LinkNet: 15-ch input (5 frames t-2..t+2 x [image, rendered hypothesis track blob (s history, d, d child), P15 detection map]),
  crop 12x32x32 pooled vox (19.5x26x26 um) centred on s/d midpoint, 12 geometry scalars; 3D ResNet-ish 2M params.
  Trained on all evaluable pairs of the training embryo (pairs from rl_cands rows: s-d, s-c, q-d, q-c; label official tp),
  hard (P15-edge FP or non-edge TP) x8 sampling weight, pos 35% of samples, 6000 steps x 512, bf16. Logit prior-corrected.
  Row score A1 = log p_sd + log(1-p_sc) + log(1-p_qd) (a-priori threshold exp(A1)>=0.5), A2 = margin.
## v1 result (held-out embryo, all sets; out/eval_v1.log)
AUC(y): ln_A1 44b6 .885 (clean .918) / 6bba .836 (clean .848) vs b1_sd .838/.794, geo-LGBM .930/.906, geo+b1 .929/.907.
-> better than b1 but clearly worse than geometry.  Precision at greedy top-K30: ln_A1 27% (44b6) / 7% (6bba); geo 23%/37%; geo+b1 27%/50%.
A-priori threshold exp(A1)>=.5: 44b6 258 applied, 42 correct, -0.0132; 6bba 597 applied, 104 correct, -0.0046.  Standalone = clear negative.
Margin score ln_A2 picks 'real' (non-flip) positives only, but precision 13-22%.
Next: 2-fold OOF CNN scores on the training embryo -> stacked LightGBM (geo+b1+CNN) LOEO to test complementarity.
## Visual check of top-scoring held-out FPs (out/mont_fp_44b6.png, out/mont_fp_6bba.png; montage.py)
- v1 is over-confident (6bba top FPs at pA ~1.00). Top-100 greedy rows: 44b6 24 TP / 76 FP, 6bba 16 TP / 84 FP.
- FPs are NOT near-duplicate matching artefacts: median d-c 7.8/8.9 um, <5% have d-c or s-q < 4 um.  Mostly dtp/dfp = -1/+1
  (the proposed relink breaks a GT-confirmed link) or 0/+1 (proposed link to a cell the GT track does not visit).
- Visually several FPs look like plausible continuations of the same nucleus (GT track jumps ~10-20 um, or GT sits on a neighbour),
  so part of the residual is annotation/matching noise, but many are plainly ambiguous dense regions; there is no clean
  image signature that separates them.
## Stacking test (stack.py; OOF 2-fold CNN scores on the training embryo, CNN trained on full training embryo for the test embryo)
train 44b6 -> test 6bba (out/stack_44b6_*.log): held-out AUC / greedy top-30 precision / train-chosen threshold result
  geo      .906 / .37 / th .6: 3 applied, +0.00002
  geob1    .907 / .50 / th .7: 2 applied, -0.00002
  geocnn   .901 / .33 / th .6: 3 applied, +0.00004
  geob1cnn .902 / .27 / th .7: 3 applied, -0.00001
  cnn      .816 / .07 / no positive threshold on train OOF
-> CNN features do not add information to geometry (AUC and top-K precision drop slightly).
train 6bba -> test 44b6 (out/stack_6bba_*.log):
  geo      .930 / .23 / th .5: 0 applied
  geob1    .929 / .27 / th .7: 0 applied
  geocnn   .931 / .23 / th .5: 0 applied
  geob1cnn .931 / .30 / th .7: 0 applied
  cnn      .880 / .20 / no positive threshold
-> in both directions no variant reaches 50% precision at 30 applied links; the CNN adds nothing over geometry.

## Official end-to-end (rule_eval.py p15 p16.linknet.ln_apply, LOEO decisions; out/rows_A1.json, out/re_*.log)
- standalone linknet, a-priori exp(A1)>=0.5: all -0.00591 CI[-0.00708,-0.00486]; 44b6 -0.01322; 6bba -0.00455; clean40 -0.00423;
  movies up 6 / down 129. strict FAIL (S1-S7).
- standalone exp(A1)>=0.9: all -0.00047 (6bba only, 55 links); FAIL.
- stacked geo+cnn, train-chosen threshold: 3 links (all 6bba) -> see out/re_geocnn.log.
- count-based deltas in evalln/stack match the official rule_eval deltas exactly (e.g. -0.01322 / -0.00455).

## Runtime (for the record; not deployable anyway)
Scoring ~7k pairs/s on RTX PRO 6000 incl. GPU crop assembly (1.05M pairs 148 s). A deployment would need to score every
candidate pair of the P-stage pool (~150k rows/movie, ~200k pairs) -> ~30 s/movie on PRO 6000, several x more on T4: far over 3 s.

## VERDICT: FAIL.  Appearance pair model beats b1 on AUC (.885/.836 vs .838/.794) but is well below geometry LightGBM (.930/.906),
is over-confident on the hard pool, and adds no complementary signal when stacked (AUC/top-K precision unchanged or lower).
The residual link errors do not carry an image signature a cross-embryo model can learn with this data; ~70% of pool positives
are matching flips and the top FPs look like ambiguous/annotation-noise cases.
- stacked geo+cnn LOEO (train-chosen thresholds): 3 links (6bba only), all +0.00004 CI[+0.00000,+0.00008], 44b6 +0.00000,
  clean40 +0.00009 -> strict FAIL (S2: 44b6 no change; far below the >=30 links/embryo bar). Same 3-link behaviour as geometry alone.
