"""Temporal-boundary combo (critic A probe): r3_tbext (edge-mode extension of tracks starting/ending within K frames of the movie
boundary via pre-ILP edges to dropped detections), then optionally r3_tbchain (reinsert dropped pre-ILP chains that touch the
boundary frame). kwargs: K, chain (dict of r3_tbchain kwargs or None)."""
WANTS_META = True
import sys


def apply(nodes, edges, K=3, chain=None, **meta):
    sys.path.insert(0, '/workspace/cl')
    from ideas import r3_tbext, r3_tbchain
    nn, ne, st = r3_tbext.apply(nodes, edges, K=K, mode='edge', **meta)
    if chain:
        nn, ne, st2 = r3_tbchain.apply(nn, ne, **dict(chain, **meta)); st.update(st2)
    return nn, ne, st
