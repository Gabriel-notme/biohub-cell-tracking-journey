#!/bin/bash
# launch all b1-recipe trainings (train_events.py unchanged) concurrently on 2 GPUs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 BLOSC_NTHREADS=1
M=/workspace/cl/p16/b1ens/models; D=/workspace/cl/p16/b1ens/deploy; i=0
run() { # data out seed
  g=$((i % 2)); i=$((i+1))
  (cd /workspace/models/b34 && CUDA_VISIBLE_DEVICES=$g setsid nohup python3 train_events.py --data $1 --out $2 --seed $3 --steps 8500 --seconds 100000 --workers 6 > $2.log 2>&1 < /dev/null &)
  echo "launched $2 seed $3 gpu $g"
}
for k in 1 2 3 4; do
  for E in 44b6 6bba; do run /dev/shm/divnet_ev_tr$E $M/loeo_tr${E}_s$k $((20260921 + k)); done
done
for k in 0 1 2; do run /dev/shm/b1ens_half$k $M/half6bba_s$k $((20260921 + k)); done
for k in 0 1 2 3 4; do run /dev/shm/b1ens_all $D/b1all_s$k $((20260921 + k)); done
