"""rule_eval module: apply precomputed (LOEO) linknet relink decisions to P15 graphs.
decisions pickle: {movie: [(s_id, d_id, c_id or -1, q_id or -1), ...]} (node ids of the P15 graph).
Action per decision: drop s->c and q->d, add s->d (skipped if the graph no longer allows it)."""
import pickle
WANTS_META = True
_CACHE = {}


def apply(nodes, edges, dec=None, name=None, **kw):
    if dec not in _CACHE: _CACHE[dec] = pickle.load(open(dec, 'rb'))
    D = _CACHE[dec].get(name, [])
    if not D: return nodes, edges, {'ln_n': 0}
    rem = set(); add = []
    for s, d, c, q in D:
        if c >= 0: rem.add((s, c))
        if q >= 0: rem.add((q, d))
        add.append((s, d))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rem]
    par = {int(e['target_id']) for e in ne}; nout = {}
    for e in ne: nout[int(e['source_id'])] = nout.get(int(e['source_id']), 0) + 1
    n = 0
    for s, d in add:
        if d in par or nout.get(s, 0) >= 1 or s not in nodes or d not in nodes: continue
        ne.append({'source_id': s, 'target_id': d, 'linknet': 1}); par.add(d); nout[s] = nout.get(s, 0) + 1; n += 1
    return nodes, ne, {'ln_n': n}
