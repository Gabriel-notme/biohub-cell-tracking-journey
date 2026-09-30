"""P20 candidate: re-apply B5's own edge veto (b2 edge probability < theta; B5 base edge_min_prob = 0.7) to edges ADDED by the
P-stage after B5 (edge_link, tb_ext, relink, long_link, tb_join, dup_join). Deletion of edges only (tracks are split; no nodes
removed). Scores precomputed on the P15 final graphs by /workspace/cl/p16/deploy/edge_b2.py."""
import json
WANTS_META = True
_S = None
ALL = ('edge_link', 'tb_ext', 'relink', 'long_link', 'tb_join', 'dup_join')


def apply(nodes, edges, theta=0.7, fams='all', model='b2', name=None, **kw):
    global _S
    if _S is None: _S = json.load(open('/workspace/cl/p16/deploy/edge_b2_scores.json'))
    fs = set(ALL if fams == 'all' else fams.split('+'))
    idx = 3 if model == 'b2' else 4
    cut = {(a, b) for a, b, k, *p in _S.get(name, []) if k in fs and p[idx - 3] < theta}
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in cut]
    return nodes, ne, {'cut': len(edges) - len(ne)}
