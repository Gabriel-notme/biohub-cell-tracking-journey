"""Rule: edge_link gap-1 candidates are generated only within RMAX=14 um. In 6bba, GT has real steps >14 um and 95 P13 FN edges are
'breaks' (matched src is a track end, matched dst a track start) with the two predicted nodes 14-25 um apart -> never candidates.
This rule re-runs the P13 edge_link model on the final graph for gap-1 end->start pairs with dmin < dist <= rmax only."""
import sys
sys.path.insert(0, '/workspace/p56stage')
import numpy as np
WANTS_META = True
_BST = {}


def apply(nodes, edges, rmax=20.0, dmin=14.0, th=0.4, need_fe=False, max_dpred=None, max_cand=None, fullgeff=None, gap=1, snap_um=3.0, **kw):
    import edge_link, lgb_np
    old = dict(edge_link.RMAX)
    try:
        if gap == 1: edge_link.RMAX[1] = float(rmax); edge_link.RMAX[2] = 1e-6
        else: edge_link.RMAX[1] = 1e-6; edge_link.RMAX[2] = float(rmax)
        rows = edge_link.features(nodes, edges, edge_link.load_full(fullgeff))
    finally:
        edge_link.RMAX.clear(); edge_link.RMAX.update(old)
    rows = [r for r in rows if r['gap'] == gap and r['dist'] > dmin]
    if gap == 2:
        return _gap2(nodes, edges, rows, th, snap_um)
    if need_fe: rows = [r for r in rows if r['fe'] > 0]
    if max_dpred is not None: rows = [r for r in rows if r['hist_s'] > 0 and r['dpred'] <= max_dpred]
    if max_cand is not None: rows = [r for r in rows if r['n_s'] <= max_cand and r['n_d'] <= max_cand]
    st = {'ll_cand': len(rows), 'll_added': 0}
    if not rows: return nodes, edges, st
    if 'm' not in _BST: _BST['m'] = lgb_np.load('/workspace/p56stage/edge_lgb.json')
    X = np.array([[r.get(k, -1) for k in edge_link.FEATS] for r in rows], dtype=np.float32)
    p = _BST['m'].predict(X)
    us, ud = set(), set(); ne = list(edges)
    for i in np.argsort(-p):
        if p[i] < th: break
        s, d = rows[i]['s'], rows[i]['d']
        if s in us or d in ud: continue
        us.add(s); ud.add(d); ne.append({'source_id': s, 'target_id': d, 'long_link': round(float(p[i]), 4)}); st['ll_added'] += 1
    return nodes, ne, st


def _gap2(nodes, edges, rows, th, snap_um):
    import edge_link, lgb_np
    st = {'ll2_cand': len(rows), 'll2_added': 0}
    if not rows: return nodes, edges, st
    if 'm' not in _BST: _BST['m'] = lgb_np.load('/workspace/p56stage/edge_lgb.json')
    X = np.array([[r.get(k, -1) for k in edge_link.FEATS] for r in rows], dtype=np.float32)
    p = _BST['m'].predict(X)
    us, ud = set(), set(); ne = list(edges); nn = dict(nodes); nid = max(int(k) for k in nodes) + 1
    for i in np.argsort(-p):
        if p[i] < th: break
        r = rows[i]; s, d = r['s'], r['d']
        if s in us or d in ud: continue
        us.add(s); ud.add(d)
        if r.get('drop_mid', 99.) <= snap_um and 'drop_xyz' in r: z, y, x = r['drop_xyz']
        else: z, y, x = [(float(nodes[s][k]) + float(nodes[d][k])) / 2 for k in 'zyx']
        nn[nid] = {'node_id': nid, 't': int(nodes[s]['t']) + 1, 'z': float(z), 'y': float(y), 'x': float(x), 'long_link_node': 1}
        ne.append({'source_id': s, 'target_id': nid, 'long_link': round(float(p[i]), 4)})
        ne.append({'source_id': nid, 'target_id': d, 'long_link': round(float(p[i]), 4)})
        nid += 1; st['ll2_added'] += 1
    return nn, ne, st
