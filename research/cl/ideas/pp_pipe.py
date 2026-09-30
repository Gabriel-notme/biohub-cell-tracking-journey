"""CPU replica of the P13 post-lineage stage for rule_eval (src p13): ignores the given P13 graph, reloads the frozen B5 graph,
re-applies div_complete with cached b1 fork probs (ideas/pp_cands, from div_rows.json), then dfork, relink, edge_link, post_prune,
with overridable parameters. With default kwargs it must reproduce P13 exactly (stat 'diff_vs_p13' = 0).
kwargs: th_start, th_stolen, K, dfork_mode ('earlier' = P13, 'later', 'b5first', 'skip_dc_in_forked'),
        relink_th, el_th, rmax1, post_prune, final_dfork (K for a dfork pass after the link stages, 0 = off)."""
import sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, '/workspace/p56stage')
WANTS_META = True
_set = set
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
CD = Path('/workspace/p56stage')


def dfork_var(nodes, edges, K=35, mode='earlier'):
    """mode 'earlier' == dfork.resolve (cut the earlier fork's other branch); 'later' cuts the later fork's farther daughter instead;
    'b5first': when exactly one of the two forks was added by div_complete, cut the div_complete fork (its added edge), else 'earlier'."""
    import numpy as np
    succ = defaultdict(list); eat = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); eat[(s, d)] = e
    forks = {n for n in nodes if len(succ.get(n, [])) >= 2}
    isdc = lambda f: any('div_complete' in eat[(f, c)] for c in succ[f])
    P = lambda n: np.array([nodes[n]['z'] * 1.625, nodes[n]['y'] * .40625, nodes[n]['x'] * .40625])
    remove = set()
    for f in sorted(forks, key=lambda n: (nodes[n]['t'], n)):
        if len(succ[f]) != 2: continue
        hit = None; f2 = None
        for c in succ[f]:
            fr = [(c, 1)]
            while fr and hit is None:
                x, dd = fr.pop()
                if x in forks: hit = c; f2 = x; break
                if dd < K:
                    for y in succ.get(x, []): fr.append((y, dd + 1))
            if hit is not None: break
        if hit is None: continue
        other = [c for c in succ[f] if c != hit][0]
        cut_later = mode == 'later' or (mode == 'b5first' and isdc(f2) and not isdc(f))
        if cut_later:
            dc_kids = [c for c in succ[f2] if 'div_complete' in eat[(f2, c)]]
            if mode == 'b5first' and dc_kids: d2 = dc_kids[0]
            else: d2 = sorted(succ[f2], key=lambda c: -np.linalg.norm(P(c) - P(f2)))[0]
            remove.add((f2, d2))
        else:
            remove.add((f, other))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in remove]
    return ne, {'dfork_removed': len(remove)}


def apply(nodes, edges, name=None, set=None, fullgeff=None, zarr=None, th_start=0.9, th_stolen=0.97, K=35, dfork_mode='earlier',
          relink_th=0.65, el_th=0.4, rmax1=None, post_prune=2, final_dfork=0, gap2=True, restore_stolen=False,
          long_r=None, long_th=None, long_model=None, long_dump=False, order='rl_first', el_twice=False, long_nomodel=None):
    import div_complete as dc, dfork, relink, edge_link, prune
    d = json.loads((Path(B5[set]) / (name + '.json')).read_text())
    n0 = {int(k): v for k, v in d['nodes'].items()}; e0 = d['edges']
    cands = json.load(open('/workspace/cl/ideas/pp_cands/%s.json' % name))
    ne, st = dc.apply(n0, e0, cands, th_start=th_start, th_stolen=th_stolen, old_max=1.0)
    if K:
        if dfork_mode == 'earlier': ne2, s4 = dfork.resolve(n0, ne, K=K)
        else: ne2, s4 = dfork_var(n0, ne, K=K, mode=dfork_mode)
        st.update(s4)
        # stats: dfork cuts of div_complete edges, and of stolen div_complete edges (whose old edge q->b is already gone)
        kept = {(int(e['source_id']), int(e['target_id'])) for e in ne2}
        cut_dc = [(int(e['source_id']), int(e['target_id'])) for e in ne if 'div_complete' in e and (int(e['source_id']), int(e['target_id'])) not in kept]
        e0s = {(int(e['source_id']), int(e['target_id'])) for e in e0}
        qof = {}
        for s_, d_ in e0s: qof[d_] = s_
        st['dfork_cut_dc'] = len(cut_dc)
        st['dfork_cut_stolen'] = sum(1 for p_, b_ in cut_dc if b_ in qof and qof[b_] != p_)
        if restore_stolen:
            outd = defaultdict(int); ind = _set()
            for s_, d_ in kept: outd[s_] += 1; ind.add(d_)
            add = []
            for p_, b_ in cut_dc:
                q_ = qof.get(b_)
                if q_ is not None and q_ != p_ and b_ not in ind and outd[q_] == 0:
                    add.append({'source_id': q_, 'target_id': b_, 'restored': 1}); outd[q_] += 1; ind.add(b_)
            ne2 = ne2 + add; st['restored'] = len(add)
        ne = ne2
    nn = n0
    full = edge_link.load_full(fullgeff)
    if relink_th is not None and relink_th < 1 and order != 'el_first':
        nn, ne, s5 = relink.apply(nn, ne, full, CD / 'relink_lgb.json', th=relink_th); st.update(s5)
    if el_th is not None and el_th < 1:
        old = dict(edge_link.RMAX)
        if rmax1: edge_link.RMAX[1] = float(rmax1)
        try:
            pre_n, pre_e = nn, ne
            nn, ne, s6 = edge_link.apply(nn, ne, fullgeff, CD / 'edge_lgb.json', th=el_th, allow_gap2=gap2); st.update(s6)
            if el_twice:
                nn, ne, s6b = edge_link.apply(nn, ne, fullgeff, CD / 'edge_lgb.json', th=el_th, allow_gap2=gap2)
                st['el2_links'] = s6b['el_gap1'] + s6b['el_gap2']
        finally:
            edge_link.RMAX.clear(); edge_link.RMAX.update(old)
        if long_r:
            # second pass: only gap-1 candidates with 14 < dist <= long_r between an end and a start that pass 1 left free.
            # Features computed on the pre-edge_link graph with RMAX[1]=long_r (same as in the rmax1=long_r variant).
            import lgb_np, numpy as np
            used_s = {int(e['source_id']) for e in ne if 'edge_link' in e}; used_d = {int(e['target_id']) for e in ne if 'edge_link' in e}
            edge_link.RMAX[1] = float(long_r)
            try:
                rows = [r for r in edge_link.features(pre_n, pre_e, full) if r['gap'] == 1 and r['dist'] > old[1]]
            finally:
                edge_link.RMAX.clear(); edge_link.RMAX.update(old)
            st['long_cands'] = len(rows); st['long_added'] = 0
            if rows:
                mp_ = ('/workspace/cl/lo/edge_base_cross_%s.json' % name[:4]) if long_model == 'cross' else (CD / 'edge_lgb.json')
                if long_nomodel is not None:  # model-free: pre-ILP candidate edge prob only
                    p = np.array([r['fe'] if r['fe'] >= long_nomodel else -1. for r in rows])
                else:
                    p = lgb_np.load(mp_).predict(np.array([[r.get(k, -1) for k in edge_link.FEATS] for r in rows], dtype=np.float32))
                has_child = {int(e['source_id']) for e in ne}; has_par = {int(e['target_id']) for e in ne}
                add = []
                for i in np.argsort(-p):
                    if p[i] < (long_th if long_th is not None else el_th): break
                    s_, d_ = rows[i]['s'], rows[i]['d']
                    if s_ in used_s or d_ in used_d or s_ in has_child or d_ in has_par: continue
                    used_s.add(s_); used_d.add(d_); has_child.add(s_); has_par.add(d_)
                    add.append({'source_id': s_, 'target_id': d_, 'edge_link': round(float(p[i]), 4), 'long': 1})
                    if long_dump: st.setdefault('long_rows', []).append(dict({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[i].items() if k != 'drop_xyz'}, p=round(float(p[i]), 3)))
                ne = ne + add; st['long_added'] = len(add)
    if relink_th is not None and relink_th < 1 and order == 'el_first':
        nn, ne, s5 = relink.apply(nn, ne, full, CD / 'relink_lgb.json', th=relink_th); st.update(s5)
    if final_dfork:
        ne, s7 = dfork.resolve(nn, ne, K=int(final_dfork)); st['final_dfork_removed'] = s7['dfork_removed']
    if post_prune:
        nn, ne, _ = prune.prune_fragments(nn, ne, int(post_prune))
    a = {(int(e['source_id']), int(e['target_id'])) for e in edges}; b = {(int(e['source_id']), int(e['target_id'])) for e in ne}
    st['diff_vs_p13'] = len(a ^ b) + abs(len(nodes) - len(nn))
    return nn, ne, st
