#!/bin/bash
set -e
cd /workspace/p12ds
for f in p_stage3.py relink.py edge_link.py lgb_np.py edge_lgb.json relink_lgb.json p5_config.json; do cp /workspace/p56stage/$f /workspace/p12ds/$f; done
python3 -m py_compile p_stage3.py relink.py edge_link.py lgb_np.py && echo compiled
python3 -c "import json,sys; [json.load(open(f)) for f in sys.argv[1:]]; print('json ok')" p5_config.json edge_lgb.json relink_lgb.json
python3 -c "import shutil; shutil.rmtree('/workspace/p12ds/__pycache__', ignore_errors=True)"
cmp /workspace/p12ds/p_stage3.py /workspace/p56stage/p_stage3.py && echo same
cat > /workspace/cl/p5_desc.md <<'EOF'
# Biohub P5 Edge Linking

Frozen B5 lineage pipeline + P3 division stage (b1 division completion, dropped-sister recovery, double-fork resolution, isolated-node pruning), followed by two new learned graph-repair steps driven by the saved pre-ILP candidate graph:

1. **Learned relinking** (LightGBM exported to JSON, numpy inference): replaces a current link with the pre-ILP candidate link when confident (th 0.65); division neighbourhoods are protected.
2. **Learned free-end linking** (LightGBM exported to JSON, numpy inference): joins a track end at t to a track start at t+1 (or t+2 through one inserted node) when confident (th 0.4).

Models were trained on 127 training movies + 32 audit movies only and validated on 36 held-out movies + the 4 preview movies with the official metric (pooled 72-movie local score 0.9726 -> 0.9786; not a leaderboard score). Every step validates graph structure and falls back per movie to the previous graph on any error.
EOF
cd /workspace/cl
python3 build_pkernel.py P5 biohub-p5-edge-linking "Biohub P5 Edge Linking" p5_config.json /workspace/kernels/p5 "$(cat /workspace/cl/p5_desc.md)" --stage-script p_stage3.py --extra-report-keys relinked,edge_linked,link_errors --record-hashes 2>&1 | tail -n 30
cat /workspace/kernels/p5/kernel-metadata.json
ls -la /workspace/p12ds
