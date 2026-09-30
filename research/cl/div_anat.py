"""Exact anatomy of P13 division errors under the official division scorer (all 199 movies).
FN GT divisions: no_window_match (parent side or <2 daughter lineages matched locally), no_local_fork, local_fork_invalid (cross/malformed),
local_fork_not_connected (fork near but daughters/timing wrong), lost_pairing. For no_local_fork: is there a pred fork on the pred lineage
through the matched parent-side node within +-4 frames (timing off by >=2)?
FP forks: which FP rule (considered = near a GT division window, evaluable, cross-component, malformed) and provenance."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']: os.environ.setdefault(_k, '1')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def job(f):
    import evalx
    import tracking_cellmot.division_metrics as dm
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    S = evalx.SCALE
    matched = dm.match_divisions(pred, gt, S, 7.)
    gdiv = dm.extract_divisions(gt)
    pred_div = {n for n in pred.node_ids() if pred.out_degree(n) >= 2}
    ev, cross, mal = dm._pred_division_fork_sets(pred, gt, S, 7.)
    invalid = cross | mal
    cands = {}; considered = set(); info = {}
    for d, mp in matched.items():
        mn = dm._matched_division_nodes(dm._matched_node_attrs(mp), gdiv[d], d)
        if mn is None: cands[d] = set(); info[d] = ('no_window_match', None); continue
        par_ids, dau_ids = mn
        local = par_ids | {s for p in par_ids for s in mp.successors(p)}
        lf = local & pred_div; considered |= lf
        good = {x for x in lf - invalid if dm._is_strongly_connected_division(mp, x, par_ids, dau_ids)}
        cands[d] = good
        if not lf: info[d] = ('no_local_fork', par_ids)
        elif not (lf - invalid): info[d] = ('local_fork_invalid', par_ids)
        elif not good: info[d] = ('local_fork_not_connected', par_ids)
        else: info[d] = ('has_candidate', par_ids)
    pairing = dm._bipartite_max_matching(list(cands), cands)
    tp = set(pairing.values()); fp = (considered | ev | invalid) - tp
    # pred lineage helpers (original ids)
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    T = {n: int(v['t']) for n, v in nodes.items()}

    def near_fork(n, k=4):  # forks on the pred lineage through n within +-k frames (walk back and forward along first child)
        out = []
        x = n
        for _ in range(k):
            if x not in par: break
            x = par[x]
            if len(succ[x]) >= 2: out.append(T[x] - T[n])
        x = n
        for _ in range(k):
            ch = succ.get(x, [])
            if len(ch) >= 2: out.append(T[x] - T[n])
            if not ch: break
            x = ch[0]
        return out
    rows = []
    for d, (cat, pids) in info.items():
        if d in pairing: rows.append(dict(kind='TP')); continue
        if cat == 'has_candidate': cat = 'lost_pairing'
        r = dict(kind='FN', cat=cat)
        if pids:
            offs = sorted({o for p in pids for o in near_fork(inv[p])})
            r['near_fork_offsets'] = offs
        rows.append(r)
    for x in fp:
        o = inv[x]
        prov = [k for k in ('relink', 'edge_link', 'xrl') if any(k in e for e in edges if int(e['source_id']) == o)]
        rows.append(dict(kind='FP', considered=int(x in considered), ev=int(x in ev), cross=int(x in cross), mal=int(x in mal),
                         t=T[o], nkids=len(succ[o]), dc=int(any(e.get('div_complete') or e.get('dc') for e in edges if int(e['source_id']) == o)),
                         edge_keys=sorted({k for e in edges if int(e['source_id']) == o for k in e if k not in ('source_id', 'target_id')})))
    return name, rows


if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(64) as p: R = p.map(job, fs)
    json.dump(dict(R), open('/workspace/cl/div_anat_p13.json', 'w'))
    for emb in ['44b6', '6bba', '']:
        rows = [r for n, rs in R if n.startswith(emb) for r in rs]
        print('==', emb or 'all', Counter(r['kind'] for r in rows))
        print('  FN cats', Counter(r['cat'] for r in rows if r['kind'] == 'FN'))
        print('  FN near-fork offsets', Counter(tuple(r.get('near_fork_offsets', ())) for r in rows if r['kind'] == 'FN').most_common(12))
        print('  FP rule flags (considered, ev, cross, mal)', Counter((r['considered'], r['ev'], r['cross'], r['mal']) for r in rows if r['kind'] == 'FP'))
        print('  FP edge keys', Counter(tuple(r['edge_keys']) for r in rows if r['kind'] == 'FP').most_common(8))
