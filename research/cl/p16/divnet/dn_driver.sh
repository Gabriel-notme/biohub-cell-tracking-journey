#!/bin/bash
# usage: dn_driver.sh <tag> <DN_MODE divnet|b1loeo> <DN_MODELS glob with {E}> <DN_MAP rank|raw> "<sets comma>"
# P15 P-stage (p15_config.json unchanged) on the original B5 lineage graphs with div_complete fork scores from a LOEO scorer (dn_patch.py).
TAG="$1"; export DN_MODE="$2"; export DN_MODELS="$3"; export DN_MAP="$4"; SETS="$5"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 BLOSC_NTHREADS=1
D=/workspace/cl/p16/divnet
python3 - <<'PY'
p = '/workspace/p56stage/p_stage9.py'; s = open(p).read()
old = "    from refine_events import EventRefiner\n    import div_complete as dc\n"
new = old + "    sys.path.insert(0, '/workspace/cl/p16/divnet'); import dn_patch; dn_patch.install(dc); print('DN_INSTALLED', os.environ.get('DN_MODE'), os.environ.get('DN_MAP'), flush=True)\n"
assert s.count(old) == 1
s = s.replace(old, new).replace("sys.path.insert(0, str(Path(__file__).parent))", "sys.path.insert(0, '/workspace/p56stage')")
s = s.replace("ngpu = max(1, min(2, torch.cuda.device_count()))", "ngpu = int(os.environ.get('DN_NW', '2'))").replace("'CUDA_VISIBLE_DEVICES': str(g)", "'CUDA_VISIBLE_DEVICES': os.environ.get('DN_GPU', str(g))")
open('/workspace/cl/p16/divnet/p_stage9dn.py', 'w').write(s)
PY
for S in ${SETS//,/ }; do
  case $S in
    prev4) RUN=/workspace/runs/b5f_prev4; FULL=/workspace/runs/fullgraph_prev4 ;;
    hold36) RUN=/workspace/runs/b5f_hold36; FULL=/workspace/runs/fullgraph_hold36 ;;
    audit32) RUN=/workspace/sync3/runs/b5f_audit32; FULL=/workspace/sync3/runs/fullgraph_audit32 ;;
    t127a) RUN=/workspace/sync4/runs/b5f_t127a; FULL=/workspace/sync4/runs/fullgraph_t127a ;;
    t127b) RUN=/workspace/sync3/runs/b5f_t127b; FULL=/workspace/sync3/runs/fullgraph_t127b ;;
  esac
  DATA=$RUN/data; [ -d "$DATA" ] || DATA=/workspace/cl/data_$S
  W=$D/ps_${TAG}_$S; mkdir -p $W
  (DN_LOG=$W/dnlog BIOHUB_BASE_REPO=$RUN/working/tracking_repo python3 $D/p_stage9dn.py --artifact /workspace/art_b56/artifact_bundle --config ${DN_CONFIG:-/workspace/p56stage/p15_config.json} --data $DATA --graphs $RUN/working/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $FULL > $W.log 2>&1; echo "PS_DONE $S $?") &
done
wait
for S in ${SETS//,/ }; do
  case $S in prev4) L=/workspace/preview4.txt ;; *) L=/workspace/$S.txt ;; esac
  W=$D/ps_${TAG}_$S
  cd /workspace/cl && python3 review.py scoreg dn_$TAG $S $W/graphs $L 2>&1 | grep -v WARN | tail -1
  grep -o '"errors": [0-9]*\|"p15_errors": [0-9]*' $W.log | tr '\n' ' '; grep -c DN_INSTALLED $W/pstage_worker_*.log | tr '\n' ' '; echo
done
echo "DN_FIN $TAG $(date -u)"
