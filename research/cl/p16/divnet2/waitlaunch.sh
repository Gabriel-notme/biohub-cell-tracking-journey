#!/bin/bash
# wait until a GPU has >= 24 GB free, then launch the given training on it
while true; do
  for g in 0 1; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    if [ "$f" -ge 24000 ]; then
      echo "launch on GPU $g free $f $(date -u)"
      CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train_v2.py "$@"
      exit
    fi
  done
  sleep 30
done
