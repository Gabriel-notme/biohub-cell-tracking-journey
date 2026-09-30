#!/bin/bash
# usage: tta_driver.sh <tag> <EVT_TTA mode> "<sets comma>" [nproc]
# re-run B5 lineage (all stages) with b1/b2 test-time augmentation, run the P15 P-stage on top, score with the official metric.
TAG="$1"; export EVT_TTA="$2"; SETS="$3"; NP=${4:-8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 BLOSC_NTHREADS=1
cd /workspace/cl && python3 lin_ablate.py $TAG none $SETS $NP > /workspace/cl/lin_$TAG.log 2>&1
unset EVT_TTA
for S in ${SETS//,/ }; do
  case $S in
    prev4) RUN=/workspace/runs/b5f_prev4; FULL=/workspace/runs/fullgraph_prev4; L=/workspace/preview4.txt ;;
    hold36) RUN=/workspace/runs/b5f_hold36; FULL=/workspace/runs/fullgraph_hold36; L=/workspace/hold36.txt ;;
    audit32) RUN=/workspace/sync3/runs/b5f_audit32; FULL=/workspace/sync3/runs/fullgraph_audit32; L=/workspace/audit32.txt ;;
    t127a) RUN=/workspace/sync4/runs/b5f_t127a; FULL=/workspace/sync4/runs/fullgraph_t127a; L=/workspace/t127a.txt ;;
    t127b) RUN=/workspace/sync3/runs/b5f_t127b; FULL=/workspace/sync3/runs/fullgraph_t127b; L=/workspace/t127b.txt ;;
  esac
  DATA=$RUN/data; [ -d "$DATA" ] || DATA=/workspace/cl/data_$S
  W=/workspace/cl/ps_${TAG}_$S; mkdir -p $W
  cd /workspace/p56stage && BIOHUB_BASE_REPO=$RUN/working/tracking_repo python3 p_stage9.py --artifact /workspace/art_b56/artifact_bundle --config /workspace/p56stage/p15_config.json --data $DATA --graphs /workspace/cl/lin/$TAG/$S/working/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $FULL > $W.log 2>&1
  cd /workspace/cl && python3 review.py scoreg $TAG $S $W/graphs $L 2>&1 | grep -v WARN | tail -1
  grep -o '"errors": [0-9]*\|"p15_errors": [0-9]*' $W.log | tr '\n' ' '; echo
done
echo "TTA_FIN $TAG $(date -u)"
