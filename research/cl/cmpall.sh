#!/bin/bash
# usage: cmpall.sh BASE cfg1 cfg2 ...
cd /workspace/cl
BASE=$1; shift
for c in "$@"; do
  echo "== $c vs $BASE"
  python cmp.py rows/${BASE}_ rows/${c}_ 2>&1 | grep -v WARN | tail -n 5
done
