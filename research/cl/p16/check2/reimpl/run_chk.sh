#!/bin/bash
# check2/reimpl: full P-stage (p_stage12.py from /workspace/p19ds, same invocation as /workspace/cl/p16/deploy/run_p19.sh) with my config
# usage: run_chk.sh <config json> <tag> "<sets comma>"
CFG="$1"; TAG="$2"; SETS="$3"
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
  W=/workspace/cl/p16/check2/reimpl/ps_${TAG}_$S
  if [ -d "$W" ]; then mv "$W" "$W.old.$$"; fi
  mkdir -p $W
  (BIOHUB_BASE_REPO=$RUN/working/tracking_repo python3 p_stage12.py --artifact /workspace/art_b56/artifact_bundle --config $CFG --data $DATA --graphs $RUN/working/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $FULL > $W.log 2>&1; echo "PS_DONE $S $?") &
done
wait
echo "CHK_FIN $TAG $(date -u)"

