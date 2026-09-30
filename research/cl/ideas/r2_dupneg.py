"""Confirmation wrapper for the duplicate-lens NEGATIVE result (r2_dup_an.py) on the real P14 graphs via rule_eval.
GT-free; reuses gen / zd_trim / drop_nodes from ideas/r2_dup_an.py unchanged (r = 3.5 um, rounded coords).
cls:
  zd     END whose nearest continuing node is 3.5-5 um away with dxy <= 1 um: drop (iterated up to 5x)
  ee_tt  END-END pairs within r: drop the shorter back-history END, then p14_post.term_trim again
  i2bub  isolated 2-node components + isolated linear 3..10-node 'bubbles' next to other components: drop
  union  ee + fd + fdb + i2 + bub drops, fsib cuts, then term_trim, then zd
"""
import sys
for p in ['/workspace/cl', '/workspace/cl/ideas', '/workspace/p56stage']:
    if p not in sys.path: sys.path.insert(0, p)
import r2_dup_an as D0


def _cut(edges, cut):
    return [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in cut]


def apply(nodes, edges, cls='zd'):
    import p14_post
    n0 = len(nodes)
    st = {}
    if cls == 'zd':
        nn, ne, cnt = D0.zd_trim(nodes, edges); st['zd'] = cnt
    else:
        D, CUT = D0.gen(nodes, edges)[:2]
        if cls == 'ee_tt':
            nn, ne = D0.drop_nodes(nodes, edges, D['ee'])
            nn, ne, s2 = p14_post.term_trim(nn, ne)
        elif cls == 'i2bub':
            nn, ne = D0.drop_nodes(nodes, edges, D['i2'] | D['bub'])
        elif cls == 'union':
            nn, ne = D0.drop_nodes(nodes, edges, D['ee'] | D['fd'] | D['fdb'] | D['i2'] | D['bub'])
            ne = _cut(ne, CUT); st['fsib_cut'] = len(CUT)
            nn, ne, s2 = p14_post.term_trim(nn, ne)
            nn, ne, cnt = D0.zd_trim(nn, ne); st['zd'] = cnt
        else:
            raise ValueError(cls)
    st['dropped'] = n0 - len(nn)
    return nn, ne, st
