# Biohub P5 Edge Linking

Frozen B5 lineage pipeline + P3 division stage (b1 division completion, dropped-sister recovery, double-fork resolution, isolated-node pruning), followed by two new learned graph-repair steps driven by the saved pre-ILP candidate graph:

1. **Learned relinking** (LightGBM exported to JSON, numpy inference): replaces a current link with the pre-ILP candidate link when confident (th 0.65); division neighbourhoods are protected.
2. **Learned free-end linking** (LightGBM exported to JSON, numpy inference): joins a track end at t to a track start at t+1 (or t+2 through one inserted node) when confident (th 0.4).

Models were trained on 127 training movies + 32 audit movies only and validated on 36 held-out movies + the 4 preview movies with the official metric (pooled 72-movie local score 0.9726 -> 0.9786; not a leaderboard score). Every step validates graph structure and falls back per movie to the previous graph on any error.
