#!/bin/bash
cd /workspace/cl
S() { [ -f "$3" ] && python review.py score $1 $2 $3 2>&1 | grep -v WARN | tail -n 1 || echo "missing $3"; }
S p3 audit32 /workspace/cl/ps_p3_audit32/submission.csv
S p5 audit32 /workspace/cl/ps_p5_audit32/submission.csv
S p4 hold36 /workspace/cl/ps_p4_hold36/submission.csv
S p4 audit32 /workspace/cl/ps_p4_audit32/submission.csv
S p6 hold36 /workspace/cl/ps_p6_hold36/submission.csv
S p6 audit32 /workspace/cl/ps_p6_audit32/submission.csv
for pair in "p3 p5" "p4 p6" "p3 p6" "p5 p6" "p3 p4"; do python review.py cmp $pair 2>&1 | grep -v WARN; done
