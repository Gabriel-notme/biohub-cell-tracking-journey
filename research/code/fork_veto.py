import sys, os, json
import numpy as np
from collections import defaultdict

def fork_probs(er, name, nodes, edges):
    from cell_event import SCALE, chain, fork_geometry
    out = defaultdict(list); prev = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); out[s].append(d); prev[d] = s
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    triples = [(s, *out[s]) for s in out if len(out[s]) == 2]
    if not triples: return []
    need = {x for t in triples for x in t}
    ids, emb = er.embeddings(name, {n: nodes[n] for n in need}); lookup = {n: i for i, n in enumerate(ids)}
    fg = [fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in triples]
    fp = er.score('fork', triples, fg, emb, lookup)
    return [{'p': p, 'a': a, 'b': b, 'fork': float(f), 'da': float(np.linalg.norm(pos[a] - pos[p])), 'db': float(np.linalg.norm(pos[b] - pos[p]))} for (p, a, b), f in zip(triples, fp)]

def apply_veto(nodes, edges, forks, veto=0.1):
    rm = set()
    for f in forks:
        if f['fork'] < veto:
            rm.add((f['p'], f['b'] if f['db'] >= f['da'] else f['a']))
    return [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm], {'vetoed': len(rm)}
