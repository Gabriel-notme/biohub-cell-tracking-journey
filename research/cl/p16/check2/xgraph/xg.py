"""chk2 xgraph: P19-R rules applied stepwise to one graph version (src); per stage: official counts (re-derived from the same
matching the official evaluate uses) + GT-level TP edge set / recovered GT division set, so TP losses can be attributed to rules.
usage: xg.py <src> [limit]   -> rows_<src>.json (rule_eval-format rows, vi=0 base, vi=1 full P19-R, vi=2.. cumulative stages)
                               + det_<src>.json (per movie per rule: removed nodes, TP GT edges lost/gained, GT divisions lost/gained)"""
import os, sys, json, glob, copy, time
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
sys.path.insert(0, '/workspace/cl/ideas'); sys.path.insert(0, '/workspace/cl/p16/deploy'); sys.path.insert(0, '/workspace/p12ds')
from pathlib import Path
from multiprocessing import Pool
import warnings; warnings.filterwarnings('ignore')
import rule_eval as RE
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
OUT = Path('/workspace/cl/p16/check2/xgraph')
STAGES = ['base', 'full', 'cd', 'ff', 'st', 'tt', 'par', 'border']  # vi 0..7 ; full must equal border (checked)
ORDER = ['base', 'cd', 'ff', 'st', 'tt', 'par', 'border']


def EK(es):
    return sorted((int(x['source_id']), int(x['target_id'])) for x in es)


def stages(nodes, edges, s, name):
    import p17_post, p19_dup, p19_edge_deploy, p14_post, chk2_xgraph_p15 as M
    refp = B5[s] + '/working/reference_graphs/%s.json' % name
    G = {'base': (nodes, edges)}
    n, e, _ = p17_post.cutdup(nodes, edges); G['cd'] = (n, e)
    n, e, _ = p17_post.forkfrag(n, e, refp); G['ff'] = (n, e)
    n, e, _ = p19_dup.start_trim(n, e, p19_dup.ref_fork_daughters(refp), r=2.5, minlen=3, iters=5); G['st'] = (n, e)
    n, e, _ = p14_post.term_trim(n, e, join=None); G['tt'] = (n, e)
    n, e, _, _ = p19_dup.par_dup(n, e, r=3.5, minrun=3, which='shorter'); G['par'] = (n, e)
    n, e, _ = p19_edge_deploy.yx_border_stubs(n, e, shape_yx=(256, 256), minlen=6, margin=2.0); G['border'] = (n, e)
    fn, fe, _ = M.apply(copy.deepcopy(nodes), copy.deepcopy(edges), name=name, set=s)  # the module under test, base kwargs
    G['full'] = (fn, fe)
    same = sorted(fn) == sorted(n) and EK(fe) == EK(e)
    return G, same


def score(nodes, edges, gt, n_total):
    import evalx
    import tracksdata as td
    from tracking_cellmot.metrics import _evaluate, _evaluate_matched_graph, per_sample_metrics, node_recall, EvaluationResult
    from tracking_cellmot.division_metrics import score_divisions
    K = td.DEFAULT_ATTR_KEYS; SC = (1.625, .40625, .40625)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    _evaluate(pred, gt, 'jaccard', SC, 7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    if pred.num_edges() == 0:
        tp = []; etp = 0; efp = 0; efn = gt.num_edges(); fpe = []
    else:
        ea = _evaluate_matched_graph(pred, gt)
        S_, D_, M_, V_ = [ea[c].to_list() for c in [K.EDGE_SOURCE, K.EDGE_TARGET, K.MATCHED_EDGE_MASK, 'pred_valid']]
        tp = sorted((p2g[inv[a]], p2g[inv[b]]) for a, b, m in zip(S_, D_, M_) if m)
        fpe = sorted((inv[a], inv[b]) for a, b, m, v in zip(S_, D_, M_, V_) if v and not m)
        etp = sum(1 for m in M_ if m); efp = sum(1 for v in V_ if v) - etp; efn = gt.num_edges() - etp
    ds = score_divisions(pred, gt, SC, 7.)
    dtp = sum(ds.scores.values())
    er = EvaluationResult(edge_tp=etp, edge_fp=efp, edge_fn=efn, division_tp=dtp, division_fp=len(ds.fp_forks),
                          division_fn=len(ds.scores) - dtp, num_pred_nodes=pred.num_nodes())
    row = per_sample_metrics(er, n_total, node_recall(pred, gt))
    det = {'tp': tp, 'fpe': fpe, 'p2g': p2g, 'gdiv_ok': sorted(int(k) for k, v in ds.scores.items() if v),
           'tpf': sorted(inv[x] for x in ds.tp_forks), 'fpf': sorted(inv[x] for x in ds.fp_forks)}
    return row, det


def job(args):
    src, s, f = args
    try:
        import numcodecs.blosc; numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx
    t0 = time.time()
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    G, same = stages(nodes, edges, s, name)
    rows = []; dets = {}; cache = {}
    for vi, st in enumerate(STAGES):
        n, e = G[st]
        key = (tuple(sorted(n)), tuple(EK(e)))
        if key in cache: r, d = cache[key]
        else: r, d = score(n, e, gt, n_total); cache[key] = (r, d)
        r = dict(r); r['movie'] = name; r['n_total'] = n_total; r['vi'] = vi; r['set'] = s; r['stage'] = st; r['same_as_module'] = same
        rows.append(r); dets[st] = d
    out = {'movie': name, 'set': s, 'same_as_module': same, 'rules': {}}
    for i in range(1, len(ORDER)):
        a, c = ORDER[i - 1], ORDER[i]
        rmv = set(G[a][0]) - set(G[c][0])
        da, dc = dets[a], dets[c]
        tpa, tpc = set(map(tuple, da['tp'])), set(map(tuple, dc['tp']))
        pa = da['p2g']; fpa = set(map(tuple, da['fpe']))
        rmE = [(x, y) for (x, y) in EK(G[a][1]) if x in rmv or y in rmv]
        rm_tp = [(x, y) for (x, y) in rmE if x in pa and y in pa and (pa[x], pa[y]) in tpa]
        rm_fp = [(x, y) for (x, y) in rmE if (x, y) in fpa]
        out['rules'][c] = {
            'n_rm': len(rmv), 'n_rm_matched': sum(1 for x in rmv if x in pa), 'n_rm_edges': len(rmE),
            'rm_tp_edges': len(rm_tp), 'rm_fp_edges': len(rm_fp), 'rm_nonvalid_edges': len(rmE) - len(rm_tp) - len(rm_fp),
            'tp_lost': len(tpa - tpc), 'tp_gained': len(tpc - tpa),
            'div_lost': sorted(set(da['gdiv_ok']) - set(dc['gdiv_ok'])), 'div_gained': sorted(set(dc['gdiv_ok']) - set(da['gdiv_ok'])),
            'fpf_removed': len(set(da['fpf']) - set(dc['fpf'])), 'fpf_added': len(set(dc['fpf']) - set(da['fpf'])),
            'tpf_removed_nodes': sorted(set(da['tpf']) & rmv),
            'rm_tp_examples': [[x, y, int(G[a][0][x]['t'])] for (x, y) in rm_tp[:30]],
            'rm_nodes': sorted(rmv)[:400]}
    b = dets['base']
    tp0, tpF = set(map(tuple, b['tp'])), set(map(tuple, dets['full']['tp']))
    out['full'] = {'tp_lost': len(tp0 - tpF), 'tp_gained': len(tpF - tp0),
                   'div_lost': sorted(set(b['gdiv_ok']) - set(dets['full']['gdiv_ok'])), 'div_gained': sorted(set(dets['full']['gdiv_ok']) - set(b['gdiv_ok'])),
                   'fpf_removed': len(set(b['fpf']) - set(dets['full']['fpf'])), 'fpf_added': len(set(dets['full']['fpf']) - set(b['fpf']))}
    out['sec'] = time.time() - t0
    return rows, out


if __name__ == '__main__':
    src = sys.argv[1]; lim = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    jobs = [(src, s, f) for s, d in RE.SRC[src].items() for f in sorted(glob.glob(d + '/*.json'))]
    if lim: jobs = jobs[:lim]
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=4) as p: R = p.map(job, jobs, chunksize=1)
    rows = [r for rs, _ in R for r in rs]; det = [d for _, d in R]
    tag = src + ('_lim%d' % lim if lim else '')
    json.dump(rows, open(OUT / ('rows_%s.json' % tag), 'w')); json.dump(det, open(OUT / ('det_%s.json' % tag), 'w'))
    print('done', src, len(det), 'movies; all same_as_module:', all(d['same_as_module'] for d in det), 'mean sec %.1f' % (sum(d['sec'] for d in det) / len(det)))
