#!/bin/bash
# usage: run_p16.sh <tag> <config file name in /workspace/p19ds> "<sets comma>"
# Full P-stage (p_stage12.py from /workspace/p19ds) on the original B5 lineage graphs, all sets in parallel, then official scoring.
TAG="$1"; CFG="$2"; SETS="$3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 BLOSC_NTHREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/p19ds
for S in ${SETS//,/ }; do
  case $S in
    prev4) RUN=/workspace/runs/b5f_prev4; FULL=/workspace/runs/fullgraph_prev4 ;;
    hold36) RUN=/workspace/runs/b5f_hold36; FULL=/workspace/runs/fullgraph_hold36 ;;
    audit32) RUN=/workspace/sync3/runs/b5f_audit32; FULL=/workspace/sync3/runs/fullgraph_audit32 ;;
    t127a) RUN=/workspace/sync4/runs/b5f_t127a; FULL=/workspace/sync4/runs/fullgraph_t127a ;;
    t127b) RUN=/workspace/sync3/runs/b5f_t127b; FULL=/workspace/sync3/runs/fullgraph_t127b ;;
  esac
  DATA=$RUN/data; [ -d "$DATA" ] || DATA=/workspace/cl/data_$S
  W=/workspace/cl/p16/ps_${TAG}_$S; rm -rf $W; mkdir -p $W
  (BIOHUB_BASE_REPO=$RUN/working/tracking_repo python3 p_stage12.py --artifact /workspace/art_b56/artifact_bundle --config /workspace/p19ds/$CFG --data $DATA --graphs $RUN/working/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $FULL > $W.log 2>&1; echo "PS_DONE $S $?") &
done
wait
for S in ${SETS//,/ }; do
  case $S in prev4) L=/workspace/preview4.txt ;; *) L=/workspace/$S.txt ;; esac
  W=/workspace/cl/p16/ps_${TAG}_$S
  cd /workspace/cl && python3 review.py scoreg $TAG $S $W/graphs $L 2>&1 | grep -v WARN | tail -1
  grep -o '"errors": [0-9]*\|"p15_errors": [0-9]*\|"seconds": [0-9.]*' $W.log | tr '\n' ' '; grep -h DIVNET16_STATS\|p1[79]_ $W/pstage_worker_*.log | cut -c1-200 | tr '\n' ' '; echo
done
echo "P19_FIN $TAG $(date -u)"
