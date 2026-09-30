#!/bin/bash
cd /workspace/cl
python review.py scoreg b3 hold36 /workspace/runs/b5_hold36/stages/04_centroid /workspace/hold36.txt 2>&1 | grep -v WARN | tail -n 1
python review.py scoreg b3 audit32 /workspace/sync3/runs/b5f_audit32/stages/04_centroid /workspace/audit32.txt 2>&1 | grep -v WARN | tail -n 1
python review.py scoreg b3 prev4 /workspace/cl/b3_prev4 /workspace/preview4.txt 2>&1 | grep -v WARN | tail -n 1
python review.py scoreg b5 hold36 /workspace/runs/b5f_hold36/working/lineage_graphs /workspace/hold36.txt 2>&1 | grep -v WARN | tail -n 1
python review.py scoreg b5 audit32 /workspace/sync3/runs/b5f_audit32/working/lineage_graphs /workspace/audit32.txt 2>&1 | grep -v WARN | tail -n 1
python review.py score b5 prev4 /workspace/kout/b5/submission.csv 2>&1 | grep -v WARN | tail -n 1
python review.py score p1 prev4 /workspace/kout/p1/submission.csv 2>&1 | grep -v WARN | tail -n 1
python review.py score p2 prev4 /workspace/kout/p2/submission.csv 2>&1 | grep -v WARN | tail -n 1
python review.py score p6kaggle prev4 /workspace/kout/p6/submission.csv 2>&1 | grep -v WARN | tail -n 1
while pgrep -f "run_pstage.sh" > /dev/null; do sleep 20; done
for c in p1 p2; do for s in hold36 audit32; do python review.py score $c $s /workspace/cl/ps_${c}_$s/submission.csv 2>&1 | grep -v WARN | tail -n 1; done; done
python review_all.py 2>&1 | grep -v WARN
