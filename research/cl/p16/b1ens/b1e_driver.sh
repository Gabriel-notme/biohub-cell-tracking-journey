#!/bin/bash
# usage: b1e_driver.sh <tag> <DN_B1LOEO glob with {E}> "<set:nworkers,...>" [44]
# P15 P-stage (p15_config.json unchanged) on the original B5 lineage graphs; div_complete fork scores from LOEO b1-recipe
# checkpoint(s) (mean logit), rank-mapped to b1dep's acceptance counts (b1e_patch.py). Optional 4th arg 44 = 44b6 movies only.
TAG="$1"; export DN_MODE=b1loeo DN_MAP=rank DN_B1LOEO="$2"; SETS="$3"; ONLY="$4"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 BLOSC_NTHREADS=1
D=/workspace/cl/p16/b1ens
for SW in ${SETS//,/ }; do
  S=${SW%%:*}; NW=${SW##*:}
  case $S in
    prev4) RUN=/workspace/runs/b5f_prev4; FULL=/workspace/runs/fullgraph_prev4 ;;
    hold36) RUN=/workspace/runs/b5f_hold36; FULL=/workspace/runs/fullgraph_hold36 ;;
    audit32) RUN=/workspace/sync3/runs/b5f_audit32; FULL=/workspace/sync3/runs/fullgraph_audit32 ;;
    t127a) RUN=/workspace/sync4/runs/b5f_t127a; FULL=/workspace/sync4/runs/fullgraph_t127a ;;
    t127b) RUN=/workspace/sync3/runs/b5f_t127b; FULL=/workspace/sync3/runs/fullgraph_t127b ;;
  esac
  if [ "$ONLY" = "44" ]; then DATA=$D/data44_$S; else DATA=$RUN/data; [ -d "$DATA" ] || DATA=/workspace/cl/data_$S; fi
  W=$D/ps_${TAG}_$S; mkdir -p $W
  (DN_NW=$NW DN_LOG=$W/dnlog BIOHUB_BASE_REPO=$RUN/working/tracking_repo python3 $D/p_stage9be.py --artifact /workspace/art_b56/artifact_bundle --config /workspace/p56stage/p15_config.json --data $DATA --graphs $RUN/working/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $FULL > $W.log 2>&1; echo "PS_DONE $S $?") &
done
wait
for SW in ${SETS//,/ }; do
  S=${SW%%:*}
  if [ "$ONLY" = "44" ]; then L=$D/${S}_44b6.txt; else case $S in prev4) L=/workspace/preview4.txt ;; *) L=/workspace/$S.txt ;; esac; fi
  W=$D/ps_${TAG}_$S
  cd /workspace/cl && python3 review.py scoreg be_$TAG $S $W/graphs $L 2>&1 | grep -v WARN | tail -1
  grep -o '"errors": [0-9]*\|"p15_errors": [0-9]*' $W.log | tr '\n' ' '; grep -h B1E_LOADED $W/pstage_worker_*.log | awk '{print $2,$3}' | sort | uniq -c | tr '\n' ' '; echo
done
echo "BE_FIN $TAG $(date -u)"
