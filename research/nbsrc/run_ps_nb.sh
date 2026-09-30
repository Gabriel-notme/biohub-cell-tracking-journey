#!/bin/bash
# run_ps_nb.sh <base variant V> <config in /workspace/p19ds> <gpu offset 0|2>: full P-stage (p_stage12_gpuoff.py = p_stage12 with worker GPU offset)
# on /workspace/nbrun/<V>_all lineage graphs (+ its reference graphs and pre-ILP fullgraphs), per set, then official scoring -> rev/nb_<V>_<cfgtag>_<set>.json
V=$1; CFG=$2; export GPU_OFFSET=$3; CT=${CFG%_config.json}; TAG=nb_${V}_${CT}
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 BLOSC_NTHREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BIOHUB_BASE_REPO=/workspace/models/support/repo
A=/workspace/nbrun/${V}_all
cd /workspace/p19ds
for S in hold36 prev4 audit32 t127a t127b; do
  W=/workspace/nbrun/ps/${TAG}_$S; mkdir -p $W
  (python3 p_stage12_gpuoff.py --artifact /workspace/art_b56/artifact_bundle --config /workspace/p19ds/$CFG --data /workspace/cl/data_$S --graphs $A/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $A/fullgraphs > $W.log 2>&1; echo "PS_DONE $S $?") &
done
wait
for S in hold36 prev4 audit32 t127a t127b; do
  case $S in prev4) L=/workspace/preview4.txt ;; *) L=/workspace/$S.txt ;; esac
  W=/workspace/nbrun/ps/${TAG}_$S
  cd /workspace/cl && python3 review.py scoreg $TAG $S $W/graphs $L 2>&1 | grep -v WARN | tail -1
  grep -o '"errors": [0-9]*\|"p15_errors": [0-9]*' $W.log | tr '\n' ' '; echo
done
echo "PSNB_FIN $TAG $(date -u)"
