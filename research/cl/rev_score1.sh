#!/bin/bash
cd /workspace/cl
ls -la /workspace/runs/b5f_hold36/working/submission.csv /workspace/kout/b5/submission.csv /workspace/sync3/runs/b5f_audit32/working/submission.csv /workspace/kout/p3v2/submission.csv /workspace/kout/p4v2/submission.csv 2>&1 | cut -c1-150
S() { [ -f "$3" ] && python review.py score $1 $2 $3 2>&1 | grep -v WARN | tail -n 1 || echo "missing $3"; }
S b5 hold36 /workspace/runs/b5f_hold36/working/submission.csv
S b5 prev4 /workspace/kout/b5/submission.csv
S b5 audit32 /workspace/sync3/runs/b5f_audit32/working/submission.csv
S p3 hold36 /workspace/cl/ps_p3_hold36/submission.csv
S p3 prev4 /workspace/kout/p3v2/submission.csv
S p4 prev4 /workspace/kout/p4v2/submission.csv
S p5 hold36 /workspace/cl/ps_p5_hold36/submission.csv
S p5 prev4 /workspace/cl/ps_p5_prev4/submission.csv
S p6 prev4 /workspace/cl/ps_p6_prev4/submission.csv
