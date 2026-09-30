#!/bin/bash
# kill the dn1 training process of embryo $1 once its seed-2 predictions exist (skip the diagnostic gt/pool arms)
while [ ! -f /workspace/cl/p16/divnet/models/dn1_full_tr$1_s2_pred.npz ]; do sleep 10; done
sleep 5; pkill -f "train_divnet.py $1 dn1" && echo "killed dn1 $1 $(date -u)"
