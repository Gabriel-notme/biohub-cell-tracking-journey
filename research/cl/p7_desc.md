# Biohub P7 Link Before Prune

P5 (frozen B5 + P3 division stage + learned relinking + learned free-end linking from the saved pre-ILP candidate graph) with one structural change: isolated-node pruning now runs AFTER the learned link steps, so free-end linking can reconnect single detections that the P5 order discarded before linking. Division decisions are identical to P5. Same LightGBM models (JSON, numpy inference). Validated on 36 held-out + 4 preview movies with the official metric (local, not leaderboard). Every step validates graph structure and falls back per movie on any error.
