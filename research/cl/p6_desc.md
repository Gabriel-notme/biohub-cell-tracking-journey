# Biohub P6 Robust Edge Linking

Frozen B5 lineage pipeline + P4 division stage (b1 division completion, double-fork resolution, isolated-node pruning; no dropped-sister recovery), followed by the two learned graph-repair steps driven by the saved pre-ILP candidate graph:

1. **Learned relinking** (LightGBM exported to JSON, numpy inference; th 0.65; division neighbourhoods protected).
2. **Learned free-end linking** (LightGBM exported to JSON, numpy inference; track end at t -> track start at t+1, or t+2 through one inserted node; th 0.4).

Models were trained on 127 training movies + 32 audit movies only and validated on 36 held-out movies + the 4 preview movies with the official metric (local, not leaderboard). Hedge of P5 on the division component. Every step validates graph structure and falls back per movie to the previous graph on any error.
