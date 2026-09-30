"""chk_ablate detail: P15 graph -> deployed P17 rules stepwise (record what each rule removes), compare with the P17 full-rerun graph,
and score P15 / P17 graphs with the official matching to get per-edge and per-fork GT status."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import importlib.util
_spec = importlib.util.spec_from_file_location('p17_post_deployed', '/workspace/p17ds/p17_post.py')
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
SETS = list(B5)
OUT = Path('/workspace/cl/p16/check/ablate/detail'); OUT.mkdir(exist_ok=True)
SC = np.array([1.625, .40625, .40625])


def sb_record(nodes, edges, maxk=2):
    out = defaultdict(list)
    for e in edges: out[int(e['source_id'])].append(int(e['target_id']))
    recs = []
    for p, ch in list(out.items()):
        if len(ch) != 2: continue
        for c in ch:
            br = [c]; n = c; ok = True
            while True:
                nx = out.get(n, [])
                if len(nx) == 0: break
                if len(nx) == 2 or len(br) > maxk: ok = False; break
                n = nx[0]; br.append(n)
            if ok and len(br) <= maxk + 1:
                recs.append({'parent': p, 'drop': br, 'keep': [x for x in ch if x != c][0]}); break
    return recs


def score_detail(nodes, edges, gt):
    import evalx
    import tracksdata as td
    from tracking_cellmot.metrics import _evaluate, _evaluate_matched_graph
    from tracking_cellmot.division_metrics import score_divisions
    K = td.DEFAULT_ATTR_KEYS; SCALE = (1.625, .40625, .40625)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    _evaluate(pred, gt, 'jaccard', SCALE, 7.)
    ea = _evaluate_matched_graph(pred, gt)
    S, D, M, V = [ea[c].to_list() for c in [K.EDGE_SOURCE, K.EDGE_TARGET, K.MATCHED_EDGE_MASK, 'pred_valid']]
    tp = sorted((inv[s], inv[d]) for s, d, m, v in zip(S, D, M, V) if m)
    fp = sorted((inv[s], inv[d]) for s, d, m, v in zip(S, D, M, V) if v and not m)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ds = score_divisions(pred, gt, SCALE, 7.)
    return {'tp': tp, 'fp': fp, 'p2g': p2g, 'tpf': sorted(inv[x] for x in ds.tp_forks), 'fpf': sorted(inv[x] for x in ds.fp_forks),
            'gscores': {int(k): int(v) for k, v in ds.scores.items()}}


def job(args):
    s, name = args
    try:
        import numcodecs.blosc; numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    n15, e15 = evalx.load_graph_json('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, name))
    n17, e17 = evalx.load_graph_json('/workspace/cl/p16/ps_p17_%s/graphs/%s.json' % (s, name))
    ps17 = json.load(open('/workspace/cl/p16/ps_p17_%s/graphs/%s.json' % (s, name))).get('pstage', {})
    refp = B5[s] + '/working/reference_graphs/%s.json' % name
    # stepwise
    na, ea, k1 = P.cutdup(n15, e15); rm_cd = sorted(set(n15) - set(na))
    nb, eb, k2 = P.forkfrag(na, ea, refp); rm_ff = sorted(set(na) - set(nb))
    sbr = sb_record(nb, eb)
    r = P.short_branch(nb, eb); nc, ec = r[0], r[1]; rm_sb = sorted(set(nb) - set(nc))
    assert sorted(x for rr in sbr for x in rr['drop']) == rm_sb
    na2, ea2, st = P.apply(n15, e15, refp)
    E = lambda es: sorted((int(e['source_id']), int(e['target_id'])) for e in es)
    same_apply = sorted(na2) == sorted(nc) and E(ea2) == E(ec)
    same_p17 = sorted(n17) == sorted(nc) and E(e17) == E(ec)
    coord_same = same_p17 and all(all(abs(float(n17[k][q]) - float(nc[k][q])) < 1e-6 for q in 'tzyx') for k in nc)
    gt, n_total = evalx.load_gt(name)
    gids = gt.node_ids(); outdeg = dict(zip(gids, gt.out_degree(gids)))
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): (int(t), np.array([z, y, x]) * SC) for i, t, z, y, x in zip(*[ga[c].to_list() for c in [K.NODE_ID, 't', 'z', 'y', 'x']])}
    gdiv_by_t = defaultdict(list)
    for g, d in outdeg.items():
        if d >= 2: gdiv_by_t[gpos[int(g)][0]].append(int(g))
    d15 = score_detail(n15, e15, gt); d17 = score_detail(n17, e17, gt)
    pos = lambda n: np.array([float(n15[n]['z']), float(n15[n]['y']), float(n15[n]['x'])]) * SC
    forks15 = defaultdict(list)
    for a, b in E(e15): forks15[a].append(b)
    nf15 = sum(1 for v in forks15.values() if len(v) == 2)
    sbinfo = []
    for rr in sbr:
        p = rr['parent']; t = int(n15[p]['t']); g = d15['p2g'].get(p)
        near = []
        for dt in (-1, 0, 1):
            for gd in gdiv_by_t.get(t + dt, []):
                dd = float(np.linalg.norm(gpos[gd][1] - pos(p)))
                if dd <= 10: near.append((dt, gd, round(dd, 2)))
        sbinfo.append(dict(rr, t=t, g=g, g_outdeg=int(outdeg.get(g, -1)) if g is not None else None, tp15=p in d15['tpf'], fp15=p in d15['fpf'],
                           fork17=len([b for a, b in E(e17) if a == p]) == 2, near_gtdiv=near,
                           drop_matched=[d15['p2g'].get(x) for x in rr['drop']], keep_matched=d15['p2g'].get(rr['keep'])))
    t15, t17 = set(map(tuple, d15['tp'])), set(map(tuple, d17['tp'])); f15, f17 = set(map(tuple, d15['fp'])), set(map(tuple, d17['fp']))
    rule_of = {**{x: 'cd' for x in rm_cd}, **{x: 'ff' for x in rm_ff}, **{x: 'sb' for x in rm_sb}}
    def rule_e(e): return rule_of.get(e[0]) or rule_of.get(e[1]) or 'other'
    res = dict(movie=name, set=s, same_apply=same_apply, same_p17=same_p17, coord_same=coord_same, ps17={k: v for k, v in ps17.items() if 'p17' in k or 'error' in k},
               n15=len(n15), n17=len(n17), rm_cd=rm_cd, rm_ff=rm_ff, rm_sb=rm_sb, nf15=nf15, sb=sbinfo,
               matched_rm={'cd': sum(1 for x in rm_cd if x in d15['p2g']), 'ff': sum(1 for x in rm_ff if x in d15['p2g']), 'sb': sum(1 for x in rm_sb if x in d15['p2g'])},
               tp_lost=[(a, b, rule_e((a, b)), d15['p2g'].get(a), d15['p2g'].get(b)) for a, b in sorted(t15 - t17)],
               tp_gained=sorted(t17 - t15), fp_removed=[(a, b, rule_e((a, b))) for a, b in sorted(f15 - f17)], fp_added=sorted(f17 - f15),
               fpf_removed=sorted(set(d15['fpf']) - set(d17['fpf'])), fpf_added=sorted(set(d17['fpf']) - set(d15['fpf'])),
               tpf_removed=sorted(set(d15['tpf']) - set(d17['tpf'])), tpf_added=sorted(set(d17['tpf']) - set(d15['tpf'])),
               cnt15=dict(tp=len(t15), fp=len(f15), tpf=len(d15['tpf']), fpf=len(d15['fpf']), dtp=sum(d15['gscores'].values())),
               cnt17=dict(tp=len(t17), fp=len(f17), tpf=len(d17['tpf']), fpf=len(d17['fpf']), dtp=sum(d17['gscores'].values())),
               n_total=n_total)
    (OUT / (name + '.json')).write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    return name


if __name__ == '__main__':
    jobs = []
    for s in SETS:
        for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s)): jobs.append((s, Path(f).stem))
    if len(sys.argv) > 1: jobs = [j for j in jobs if j[1] in sys.argv[1:]]
    with Pool(int(os.environ.get('DPOOL', '3')), maxtasksperchild=4) as p:
        for i, n in enumerate(p.imap_unordered(job, jobs)): print(i, n, flush=True)
