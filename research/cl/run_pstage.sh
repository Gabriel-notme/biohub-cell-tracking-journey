#!/bin/bash
# usage: run_pstage.sh <stage_dir> <config.json> <TAG: prev4|hold36|audit32> <outname>
SD=$1; CFG=$2; TAG=$3; OUTN=$4
case $TAG in
  prev4) RUN=/workspace/runs/b5f_prev4; FULL=/workspace/runs/fullgraph_prev4 ;;
  hold36) RUN=/workspace/runs/b5f_hold36; FULL=/workspace/runs/fullgraph_hold36 ;;
  dtrain) RUN=/workspace/runs/b5d_train; FULL=/workspace/runs/fg2_train ;;
  dval) RUN=/workspace/runs/b5d_val; FULL=/workspace/runs/fg2_val ;;
  audit32) RUN=/workspace/sync3/runs/b5f_audit32; FULL=/workspace/sync3/runs/fullgraph_audit32 ;;
  t127a) RUN=/workspace/sync4/runs/b5f_t127a; FULL=/workspace/sync4/runs/fullgraph_t127a ;;
  t127b) RUN=/workspace/sync3/runs/b5f_t127b; FULL=/workspace/sync3/runs/fullgraph_t127b ;;
esac
DATA=$RUN/data
[ -d "$DATA" ] || { mkdir -p /workspace/cl/data_$TAG; for m in $(cat /workspace/$( [ $TAG = prev4 ] && echo preview4 || echo $TAG ).txt); do ln -sfn /workspace/data/train/$m.zarr /workspace/cl/data_$TAG/$m.zarr; done; DATA=/workspace/cl/data_$TAG; }
W=/workspace/cl/ps_${OUTN}_$TAG
rm -rf $W; mkdir -p $W
cd $SD
BIOHUB_BASE_REPO=$RUN/working/tracking_repo python ${SCRIPT:-p_stage3.py} --artifact /workspace/art_b56/artifact_bundle --config $CFG --data $DATA --graphs $RUN/working/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $FULL > $W.log 2>&1
grep PSTAGE_EXPORT_COMPLETE $W.log | cut -c1-600
grep -h 'PSTAGE_MOVIE' $W/pstage_worker_*.log | head -n 3 | cut -c1-400
