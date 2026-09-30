#!/bin/bash
# usage: ptta_driver.sh <tag> <EVT_TTA mode> "<sets comma>"
# P15 P-stage on the original B5 lineage graphs, with b1/b2 test-time augmentation only inside the P-stage (division completion
# scoring), all sets in parallel; then score every set with the official metric.
TAG="$1"; export EVT_TTA="$2"; SETS="$3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 BLOSC_NTHREADS=1
cd /workspace/p56stage
python3 - <<'EOF'
p = '/workspace/p56stage/p_stage9.py'; s = open(p).read()
old = "    from refine_events import EventRefiner\n    import div_complete as dc\n"
new = old + "    if os.environ.get('EVT_TTA', 'none') != 'none':\n        sys.path.insert(0, '/workspace/cl'); import evt_tta; evt_tta.install(); print('EVT_TTA_INSTALLED', os.environ['EVT_TTA'], flush=True)\n"
assert s.count(old) == 1; open('/workspace/p56stage/p_stage9t.py', 'w').write(s.replace(old, new))
EOF
for S in ${SETS//,/ }; do
  case $S in
    prev4) RUN=/workspace/runs/b5f_prev4; FULL=/workspace/runs/fullgraph_prev4 ;;
    hold36) RUN=/workspace/runs/b5f_hold36; FULL=/workspace/runs/fullgraph_hold36 ;;
    audit32) RUN=/workspace/sync3/runs/b5f_audit32; FULL=/workspace/sync3/runs/fullgraph_audit32 ;;
    t127a) RUN=/workspace/sync4/runs/b5f_t127a; FULL=/workspace/sync4/runs/fullgraph_t127a ;;
    t127b) RUN=/workspace/sync3/runs/b5f_t127b; FULL=/workspace/sync3/runs/fullgraph_t127b ;;
  esac
  DATA=$RUN/data; [ -d "$DATA" ] || DATA=/workspace/cl/data_$S
  W=/workspace/cl/ps_${TAG}_$S; mkdir -p $W
  (BIOHUB_BASE_REPO=$RUN/working/tracking_repo python3 p_stage9t.py --artifact /workspace/art_b56/artifact_bundle --config /workspace/p56stage/p15_config.json --data $DATA --graphs $RUN/working/lineage_graphs --out $W/graphs --work $W --submission $W/submission.csv --full $FULL > $W.log 2>&1; echo "PS_DONE $S $?") &
done
wait
unset EVT_TTA
for S in ${SETS//,/ }; do
  case $S in prev4) L=/workspace/preview4.txt ;; *) L=/workspace/$S.txt ;; esac
  W=/workspace/cl/ps_${TAG}_$S
  cd /workspace/cl && python3 review.py scoreg $TAG $S $W/graphs $L 2>&1 | grep -v WARN | tail -1
  grep -o '"errors": [0-9]*\|"p15_errors": [0-9]*' $W.log | tr '\n' ' '; grep -c EVT_TTA_INSTALLED $W/pstage_worker_*.log | tr '\n' ' '; echo
done
echo "PTTA_FIN $TAG $(date -u)"
