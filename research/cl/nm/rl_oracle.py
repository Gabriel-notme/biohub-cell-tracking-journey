"""Link-stage oracle on P15 graphs (all 199 movies). For every GT edge that is FN although BOTH endpoints are matched,
build the link action (add p1->p2, drop the non-TP edges that block it) and classify it; apply oracle subsets and score
with the official metric.  Also labels the whole pre-ILP candidate pool (pre-ILP edges absent from P15) by structure type.
usage: rl_oracle.py  -> /workspace/cl/nm/rl_oracle/<set>__<movie>.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
OUT = Path('/workspace/cl/nm/rl_oracle'); OUT.mkdir(exist_ok=True, parents=True)
PROV = ('relink', 'edge_link', 'long_link', 'tb_ext', 'tb_join', 'div_complete', 'dup_join')


def apply_actions(nodes, edges, adds, rems):
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rems]
    have = {(int(e['source_id']), int(e['target_id'])) for e in ne}
    ne += [{'source_id': a, 'target_id': b, 'oracle': 1} for a, b in adds if (a, b) not in have]
    # enforce constraints (in<=1, out<=2): later adds lose
    par = {}; out = Counter(); keep = []
    for e in ne:
        a, b = int(e['source_id']), int(e['target_id'])
        if b in par or out[a] >= 2: continue
        par[b] = a; out[a] += 1; keep.append(e)
    return nodes, keep


def job(a):
    s, f = a
    name = Path(f).stem; of = OUT / ('%s__%s.json' % (s, name))
    if of.exists(): return 1
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import evalx, edge_link
    from loeo import gt_maps
    nodes, edges = evalx.load_graph_json(f)
    fids, fT, fV, fE, fprob = edge_link.load_full(FULL[s] + '/' + name + '.geff')
    fedge = {(int(x), int(y)): float(p) for (x, y), p in zip(fE.tolist(), fprob.tolist())}
    p2g, gs, gp = gt_maps(name, nodes, edges)
    g2p = {g: p for p, g in p2g.items()}
    gt, _ = evalx.load_gt(name); ga = gt.node_attrs(attr_keys=[evalx.K.NODE_ID, 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) * S for i, z, y, x in zip(*[ga[k].to_list() for k in [evalx.K.NODE_ID, 'z', 'y', 'x']])}
    succ = defaultdict(list); par = {}; prov = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); succ[x].append(y); par[y] = x
        prov[(x, y)] = next((k for k in PROV if k in e), 'b5')
    pos = {n: np.array([v['z'], v['y'], v['x']], float) * S for n, v in nodes.items()}

    def valid(x, y):
        gx, gy = p2g.get(x), p2g.get(y)
        return (gx is not None and len(gs.get(gx, [])) > 0) or (gy is not None and gy in gp)

    def tp(x, y):
        gx, gy = p2g.get(x), p2g.get(y)
        return gx is not None and gy is not None and gy in gs.get(gx, [])

    def lab(x, y):
        return 'TP' if tp(x, y) else ('FP' if valid(x, y) else 'NE')
    cnt = Counter(lab(int(e['source_id']), int(e['target_id'])) for e in edges)
    provlab = Counter('%s|%s' % (prov[k], lab(*k)) for k in prov)
    # FN GT edges
    fixes = []; fncat = Counter()
    for g1, ch in gs.items():
        for g2 in ch:
            p1, p2 = g2p.get(g1), g2p.get(g2)
            if p1 is None or p2 is None:
                fncat['node_missing'] += 1; continue
            if p2 in succ.get(p1, []): continue
            kids = succ.get(p1, []); q = par.get(p2)
            rm = []
            for c in kids:
                if not tp(p1, c): rm.append((p1, c))
            if q is not None: rm.append((q, p2))
            ktp = [c for c in kids if tp(p1, c)]
            typ = ('ff' if (not kids and q is None) else 'end_taken' if not kids else 'start_else' if q is None else 'swap')
            if ktp: typ = 'div_' + typ  # p1 already holds a TP child: this is a missing division daughter
            fe = fedge.get((p1, p2), -1.)
            rml = [lab(*r) for r in rm]
            dcg = min([float(np.linalg.norm(pos[c] - gpos[g2])) for c in kids if not tp(p1, c)] or [-1.])
            dqg = float(np.linalg.norm(pos[q] - gpos[g1])) if q is not None else -1.
            dpg = [round(float(np.linalg.norm(pos[p1] - gpos[g1])), 2), round(float(np.linalg.norm(pos[p2] - gpos[g2])), 2)]
            fixes.append(dict(p1=p1, p2=p2, t=int(nodes[p1]['t']), typ=typ, fe=round(fe, 4), dist=round(float(np.linalg.norm(pos[p2] - pos[p1])), 2),
                              rm=[list(r) for r in rm], rml=rml, rmprov=[prov[r] for r in rm], gdiv=int(len(ch) == 2), nkids=len(kids), hasq=int(q is not None), dcg=round(dcg, 2), dqg=round(dqg, 2), dpg=dpg,
                              flip=int((0 <= dcg < 7) or (0 <= dqg < 7))))
            fncat[typ + ('|pre' if fe >= 0 else '|nopre')] += 1
    # pre-ILP candidate pool (edges absent from P15) by structure type
    pool = Counter()
    for (x, y), p in fedge.items():
        if x not in nodes or y not in nodes or y in succ.get(x, []): continue
        if int(nodes[y]['t']) != int(nodes[x]['t']) + 1: continue
        kids = succ.get(x, []); q = par.get(y)
        typ = ('ff' if (not kids and q is None) else 'end_taken' if not kids else 'start_else' if q is None else 'swap')
        if len(kids) >= 2: typ = 'forkpar_' + typ
        l = lab(x, y); pb = 'p>=.5' if p >= 0.5 else ('p>=.2' if p >= 0.2 else 'p<.2')
        pool['%s|%s|%s' % (typ, l, pb)] += 1
    # ff spatial pool beyond pre-ILP (ends at t -> starts at t+1 within 20 um)
    from scipy.spatial import cKDTree
    starts = defaultdict(list)
    for n in nodes:
        if n not in par: starts[int(nodes[n]['t'])].append(n)
    st = {t: (ns, cKDTree(np.array([pos[n] for n in ns]))) for t, ns in starts.items()}
    for n in nodes:
        if succ.get(n): continue
        t = int(nodes[n]['t'])
        if t + 1 not in st: continue
        ns, tr = st[t + 1]
        for j in tr.query_ball_point(pos[n], 20.0):
            y = ns[j]; d = float(np.linalg.norm(pos[y] - pos[n]))
            pool['ffsp|%s|%s|%s' % ('pre' if (n, y) in fedge else 'nopre', lab(n, y), '<7' if d < 7 else ('<14' if d < 14 else '<20'))] += 1
    # oracle variants
    def sel(pred):
        adds, rems = [], set()
        for fx in fixes:
            if 'TP' in fx['rml']: continue
            if pred(fx):
                adds.append((fx['p1'], fx['p2'])); rems.update(tuple(r) for r in fx['rm'])
        return adds, rems
    V = {'base': ([], set()),
         'ff_pre': sel(lambda fx: fx['typ'] == 'ff' and fx['fe'] >= 0),
         'ff_all': sel(lambda fx: fx['typ'] == 'ff'),
         'rl_pre': sel(lambda fx: fx['typ'] in ('end_taken', 'start_else', 'swap') and fx['fe'] >= 0),
         'rl_all': sel(lambda fx: fx['typ'] in ('end_taken', 'start_else', 'swap')),
         'div': sel(lambda fx: fx['typ'].startswith('div_')),
         'link_all': sel(lambda fx: not fx['typ'].startswith('div_')),
         'link_pre': sel(lambda fx: not fx['typ'].startswith('div_') and fx['fe'] >= 0),
         'link_noflip': sel(lambda fx: not fx['typ'].startswith('div_') and not fx['flip']),
         'link_pre_noflip': sel(lambda fx: not fx['typ'].startswith('div_') and fx['fe'] >= 0 and not fx['flip'])}
    revs = {k for k, v in prov.items() if v in ('relink', 'edge_link', 'long_link', 'tb_join') and lab(*k) == 'FP'}
    V['rev_fp_links'] = ([], revs)
    a_, r_ = V['link_all']; V['link_all+rev'] = (a_, r_ | revs)
    scores = {}
    for k, (adds, rems) in V.items():
        nn, ee = apply_actions(nodes, edges, adds, rems)
        r = evalx.score_movie(name, nn, ee); r['n_add'] = len(adds); r['n_rm'] = len(rems); scores[k] = r
    of.write_text(json.dumps(dict(movie=name, set=s, cnt=cnt, provlab=provlab, fncat=fncat, fixes=fixes, pool=pool, scores=scores)))
    return 1


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(int(os.environ.get('NPROC', '20')), maxtasksperchild=4) as p: print('done', sum(p.map(job, jobs, chunksize=1)), len(jobs))
