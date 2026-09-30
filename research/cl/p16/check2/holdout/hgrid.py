"""check2/holdout: P19-R parameter grid (one-at-a-time around the default, family-off options, arbitrary extra configs) on the P15
graphs of all 199 movies, official per-movie scoring, deduped by final node set; plus per-item records of the default config.
usage: hgrid.py <configs.json> <outdir> [items]"""
import os, sys, json, glob, time, traceback
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path[:0] = ['/workspace/cl', '/workspace/official/src', '/workspace/code', '/workspace/cl/ideas', '/workspace/cl/p16/deploy', '/workspace/p12ds']
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import warnings; warnings.filterwarnings('ignore')
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
SETS = list(B5)
SC = np.array([1.625, .40625, .40625])
DEF = dict(cd=3.2, ff=6, st=2.5, tt=3.5, par=3.5, bm=2.0, bl=6)
ORDER = ['cd', 'ff', 'st', 'tt', 'par', 'bd']


def sub(nodes, edges, drop):
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop])


def comps_of(nset, edges):
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in nset and b in nset: adj[a].append(b); adj[b].append(a)
    seen = set(); out = []
    for n in sorted(nset):
        if n in seen: continue
        c = []; st = [n]; seen.add(n)
        while st:
            u = st.pop(); c.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); st.append(v)
        out.append(sorted(c))
    return out


def step(fam, nodes, edges, refp, c):
    import p17_post, p19_dup, p14_post, p19_edge_deploy
    if fam == 'cd':
        if c['cd'] is None: return nodes, edges
        r = p17_post.cutdup(nodes, edges, rad=c['cd'])
    elif fam == 'ff':
        if c['ff'] is None: return nodes, edges
        r = p17_post.forkfrag(nodes, edges, refp, minlen=c['ff'])
    elif fam == 'st':
        if c['st'] is None: return nodes, edges
        r = p19_dup.start_trim(nodes, edges, p19_dup.ref_fork_daughters(refp), r=c['st'], minlen=3, iters=5)
    elif fam == 'tt':
        if c['tt'] is None: return nodes, edges
        r = p14_post.term_trim(nodes, edges, r=c['tt'], join=None)
    elif fam == 'par':
        if c['par'] is None: return nodes, edges
        r = p19_dup.par_dup(nodes, edges, r=c['par'], minrun=3, which='shorter')
    elif fam == 'bd':
        if c['bm'] is None or c['bl'] is None: return nodes, edges
        r = p19_edge_deploy.yx_border_stubs(nodes, edges, shape_yx=(256, 256), minlen=c['bl'], margin=c['bm'])
    return r[0], r[1]


def keyof(fam, c):
    return {'cd': c['cd'], 'ff': c['ff'], 'st': c['st'], 'tt': c['tt'], 'par': c['par'],
            'bd': (c['bm'], c['bl']) if c['bm'] is not None and c['bl'] is not None else None}[fam]


def score_detail(nodes, edges, gt):
    import evalx
    import tracksdata as td
    from tracking_cellmot.metrics import _evaluate, _evaluate_matched_graph
    K = td.DEFAULT_ATTR_KEYS; SCALE = (1.625, .40625, .40625)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    _evaluate(pred, gt, 'jaccard', SCALE, 7.)
    ea = _evaluate_matched_graph(pred, gt)
    S, D, M, V = [ea[c].to_list() for c in [K.EDGE_SOURCE, K.EDGE_TARGET, K.MATCHED_EDGE_MASK, 'pred_valid']]
    tp = {(inv[s], inv[d]) for s, d, m, v in zip(S, D, M, V) if m}
    fp = {(inv[s], inv[d]) for s, d, m, v in zip(S, D, M, V) if v and not m}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    return tp, fp, p2g


def job(args):
    s, f, cfgs, outdir, do_items = args
    name = os.path.basename(f)[:-5]; op = '%s/%s.json' % (outdir, name)
    if os.path.exists(op): return name, 'cached'
    try:
        try:
            import numcodecs.blosc; numcodecs.blosc.use_threads = False
        except Exception:
            pass
        import evalx
        t0 = time.time()
        refp = B5[s] + '/working/reference_graphs/%s.json' % name
        n0, e0 = evalx.load_graph_json(f)
        cache = {(): (n0, e0)}
        famdel = []; finals = []
        for c in cfgs:
            pre = (); nodes, edges = n0, e0; fd = {}
            for fam in ORDER:
                k = pre + ((fam, keyof(fam, c)),)
                if k not in cache: cache[k] = step(fam, nodes, edges, refp, c)
                nn, ee = cache[k]; fd['d_' + fam] = len(nodes) - len(nn); nodes, edges = nn, ee; pre = k
            famdel.append(fd); finals.append(frozenset(nodes))
        uniq = {}; rows = []
        base = evalx.score_movie(name, n0, e0); base.update(vi=0, set=s); rows.append(base)
        uniq[frozenset(n0)] = base
        for i, (c, fs) in enumerate(zip(cfgs, finals), 1):
            if fs not in uniq:
                nn, ee = sub(n0, e0, set(n0) - fs); uniq[fs] = evalx.score_movie(name, nn, ee)
            r = dict(uniq[fs]); r.update(famdel[i - 1]); r.update(vi=i, set=s, movie=name); rows.append(r)
        res = {'movie': name, 'set': s, 'rows': rows, 'n_uniq': len(uniq)}
        if do_items:  # per-item records of cfg 1 (the default), relative to the P15 graph
            gt, n_total = evalx.load_gt(name)
            tp, fp, p2g = score_detail(n0, e0, gt)
            inc = defaultdict(list)
            for e in e0:
                a, b = int(e['source_id']), int(e['target_id']); inc[a].append((a, b)); inc[b].append((a, b))
            P = {u: np.array([float(n0[u][q]) for q in 'zyx']) * SC for u in n0}
            byt = defaultdict(list)
            for u, v in n0.items(): byt[int(v['t'])].append(u)
            c = cfgs[0]; pre = (); nodes, edges = n0, e0; items = []
            for fam in ORDER:
                k = pre + ((fam, keyof(fam, c)),); nn, ee = cache[k]
                gone = set(nodes) - set(nn)
                for comp in comps_of(gone, edges):
                    ie = {x for u in comp for x in inc[u]}
                    it = {'fam': fam, 'nodes': comp, 'matched': {u: p2g[u] for u in comp if u in p2g},
                          'tp_inc': sorted(x for x in ie if x in tp), 'fp_inc': sorted(x for x in ie if x in fp)}
                    it['touch'] = bool(it['matched'] or it['tp_inc'] or it['fp_inc'])
                    it['pos'] = [[int(n0[u]['t'])] + [round(float(n0[u][q]), 1) for q in 'zyx'] for u in comp]
                    items.append(it)
                nodes, edges = nn, ee; pre = k
            for it in items:
                if not it['touch']: continue
                nn, ee = sub(n0, e0, set(it['nodes']))
                r = evalx.score_movie(name, nn, ee)
                it['marg'] = {q: r[q] - base[q] for q in ['edge_tp', 'edge_fp', 'edge_fn', 'division_tp', 'division_fp', 'division_fn', 'num_pred_nodes']}
                tp2, fp2, p2g2 = score_detail(nn, ee, gt)
                g2p2 = {g: p for p, g in p2g2.items()}
                it['rematch'] = {str(u): g2p2.get(g) for u, g in it['matched'].items()}
                it['tp_new'] = sorted(tp2 - tp); it['fp_new'] = sorted(fp2 - fp)
                itn = set(it['nodes'])
                it['nn_um'] = [round(min([float(np.linalg.norm(P[u] - P[w])) for w in byt[int(n0[u]['t'])] if w != u and w not in itn] or [99.]), 2) for u in it['nodes']]
            res['items'] = items
            res['n_items'] = {fam: sum(1 for it in items if it['fam'] == fam) for fam in ORDER}
        res['sec'] = time.time() - t0
        json.dump(res, open(op + '.tmp', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o)); os.replace(op + '.tmp', op)
        return name, 'ok %.0fs uniq %d' % (res['sec'], len(uniq))
    except Exception:
        return name, 'ERR ' + traceback.format_exc()[-2000:]


if __name__ == '__main__':
    cfgs = json.load(open(sys.argv[1])); outdir = sys.argv[2]; do_items = len(sys.argv) > 3 and sys.argv[3] == 'items'
    flt = os.environ.get('FLT', '')
    os.makedirs(outdir, exist_ok=True)
    jobs = [(s, f, cfgs, outdir, do_items) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s)) if flt in os.path.basename(f)]
    print(len(jobs), 'jobs', len(cfgs), 'cfgs', flush=True)
    if 'SHARD' in os.environ:  # sequential shard k/n in this process (no fork)
        k, n = map(int, os.environ['SHARD'].split('/'))
        for j in jobs[k::n]: print(*job(j), flush=True)
    else:
        with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=4) as p:
            for name, st in p.imap_unordered(job, jobs, chunksize=1): print(name, st, flush=True)
