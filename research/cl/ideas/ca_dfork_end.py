"""Rule: dfork runs BEFORE relink/edge_link in p_stage4, so double forks created later by linking are never resolved
(13 lineage double-fork pairs remain in P13 output; 2 within K=35 were created by relink/edge_link). Re-apply dfork at the end."""
import sys
sys.path.insert(0, '/workspace/p56stage')


def apply(nodes, edges, K=35, prune=True):
    import dfork, prune as pr
    ne, st = dfork.resolve(nodes, edges, K=int(K))
    nn = nodes
    if prune: nn, ne, _ = pr.prune_fragments(nodes, ne, 2)
    return nn, ne, st
