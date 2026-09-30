"""dv_start_rule: apply the LOEO learned start-type division completion to a P15 graph (scores precomputed by dv_train_rule.py with
the model of the OTHER embryo; threshold chosen on that other embryo). Greedy: one fork per p and per b; adds edge p->b (b is a track start).
Only evaluable candidates were scored (non-evaluable additions do not change division counts). kw: r (adds-per-movie level)."""
import json
WANTS_META = True
_S = None; _T = None


def apply(nodes, edges, r='0.1', name=None, **kw):
    global _S, _T
    if _S is None:
        _S = json.load(open('/workspace/cl/nm/dv_rule_scores.json')); _T = json.load(open('/workspace/cl/nm/dv_rule_thr.json'))
    other = '6bba' if name.startswith('44b6') else '44b6'
    thr = _T[other][r]
    has_par = {int(e['target_id']) for e in edges}
    nout = {}
    for e in edges: nout[int(e['source_id'])] = nout.get(int(e['source_id']), 0) + 1
    cs = sorted(_S.get(name, []), key=lambda x: -x[2])
    used_p, used_b, add = set(), set(), []
    for p, b, s, lab, st in cs:
        if s < thr: break
        if p in used_p or b in used_b or b in has_par or nout.get(p, 0) != 1: continue
        used_p.add(p); used_b.add(b); add.append((p, b))
    ne = list(edges) + [{'source_id': p, 'target_id': b, 'div_learned': 1} for p, b in add]
    return nodes, ne, {'dl_added': len(add)}
