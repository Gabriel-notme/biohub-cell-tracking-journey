"""Fork veto on P15 graphs: remove a fork F->(c1,c2) whose LOEO DivNet score (score_forks.py output, env DN_FORKS) is < th by cutting
the edge to the child with the shorter forward chain (tie: the child farther from F). For cl/rule_eval.py p15 p16.divnet.dn_veto '[{"th":..}]'."""
import os, json
from collections import defaultdict
import numpy as np
_S = {}


def apply(nodes, edges, th=0.05, **kw):
    if 'S' not in _S: _S['S'] = json.load(open(os.environ.get('DN_FORKS', '/workspace/cl/p16/divnet/fork_scores_dn1.json')))
    name = kw.get('name')
    if name is None:
        return nodes, edges, {'veto': 0}
    sc = _S['S'].get(name, {})
    out = defaultdict(list)
    for e in edges: out[int(e['source_id'])].append(int(e['target_id']))

    def flen(n, k=30):
        c = 0
        while c < k and len(out.get(n, [])) == 1: n = out[n][0]; c += 1
        return c
    rm = set()
    for p, ch in out.items():
        if len(ch) != 2 or str(p) not in sc or sc[str(p)] >= th: continue
        P = np.array([nodes[p][k] for k in 'zyx']) * [1.625, .40625, .40625]
        key = lambda c: (flen(c), -np.linalg.norm(np.array([nodes[c][k] for k in 'zyx']) * [1.625, .40625, .40625] - P))
        rm.add((p, min(ch, key=key)))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm]
    return nodes, ne, {'veto': len(rm)}


WANTS_META = True
