"""Division-lens probe: veto div_complete forks whose continuation edge p->a was itself created by a B5 post-hoc repair
(joint_event = new link from B5 joint refinement; learned_recovery = link through interpolated nodes; gap_closed / gap2_recovered).
Removes the div_complete edge p->b; optionally restores the B5 edge q->b when b was stolen from q and q is still a childless end."""
from collections import defaultdict
import json
WANTS_META = True
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def apply(nodes, edges, flags=('joint_event', 'learned_recovery'), restore=False, **kw):
    ch = defaultdict(list); E = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); E[(u, v)] = e
    bpar = None
    if restore:
        d = json.load(open('%s/%s.json' % (B5[kw['set']], kw['name'])))
        bpar = {int(e['target_id']): int(e['source_id']) for e in d['edges']}
    rm = []; add = []
    for p, ks in ch.items():
        if len(ks) != 2: continue
        dcs = [k for k in ks if 'div_complete' in E[(p, k)]]
        if len(dcs) != 1: continue
        b = dcs[0]; a = [k for k in ks if k != b][0]
        if not any(f in E[(p, a)] for f in flags): continue
        rm.append((p, b))
        if restore:
            q = bpar.get(b)
            if q is not None and q != p and q in nodes and not ch.get(q) and int(nodes[q]['t']) + 1 == int(nodes[b]['t']):
                add.append({'source_id': q, 'target_id': b})
    rs = frozenset(rm)
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rs] + add
    return nodes, ne, {'jv_removed': len(rm), 'jv_restored': len(add)}
