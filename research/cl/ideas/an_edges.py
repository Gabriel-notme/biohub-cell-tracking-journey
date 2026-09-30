"""Per-edge anatomy of P13 graphs vs GT (analysis only). Dumps evaluable pred edges + FN GT edges with structural features."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS']: os.environ[_k]='1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
S = np.array([1.625, 0.40625, 0.40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']

def etype(e):
    for k in ['gap_closed','edge_link','gap2_recovered','joint_event','relink','div_complete','safe_division','event_division','event_edge','learned_recovery']:
        if k in e: return k
    return e.get('motion_pass', 'other')

def job(f):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {v: k for k, v in p2g.items()}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = defaultdict(list); GE = set()
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()):
        gs[int(x)].append(int(y)); gp[int(y)].append(int(x)); GE.add((int(x), int(y)))
    gat = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(gat[K.NODE_ID].to_list(), gat['z'].to_list(), gat['y'].to_list(), gat['x'].to_list())}
    gtt = dict(zip([int(i) for i in gat[K.NODE_ID].to_list()], gat['t'].to_list()))
    out = defaultdict(list); par = {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); par[y] = x
    # rounded positions as the metric sees them
    pos = {n: np.array([max(0, int(round(v[k]))) for k in 'zyx']) * S for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    trees = {t: (cKDTree(np.stack([pos[n] for n in ns])), ns) for t, ns in byt.items()}
    def nn2(n):
        t = int(nodes[n]['t']); tr, ns = trees[t]
        d, i = tr.query(pos[n], k=min(3, len(ns)))
        d = np.atleast_1d(d); i = np.atleast_1d(i)
        oth = [(dd, ns[ii]) for dd, ii in zip(d, i) if ns[ii] != n]
        return oth[0] if oth else (99., None)
    # segment length: number of nodes in the linear run containing n (between forks/ends)
    def run_len(n):
        L = 1; a = n
        while a in par and len(out[par[a]]) == 1 and len(out[a]) <= 1: a = par[a]; L += 1
        b = n
        while len(out[b]) == 1: b = out[b][0]; L += 1
        return L
    def ntype(n):
        v = nodes[n]
        return 'gs' if 'gap_synthetic' in v else ('el' if 'edge_link_node' in v else ('lr' if 'learned_recovery' in v else 'n'))
    rows = []
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id'])
        gx, gy = p2g.get(x), p2g.get(y)
        ev = bool((gx is not None and gs.get(gx)) or (gy is not None and gp.get(gy)))
        tp = gx is not None and gy is not None and (gx, gy) in GE
        lab = 'TP' if tp else ('FP' if ev else 'NE')
        d1, nb1 = nn2(x); d2, nb2 = nn2(y)
        r = dict(m=name, set=st, lab=lab, et=etype(e), p=float(e.get('edge_prob') if e.get('edge_prob') is not None else -1), disp=float(np.linalg.norm(pos[y] - pos[x])),
                 sfork=len(out[x]) == 2, dfork=len(out[y]) == 2, sstart=x not in par, dend=len(out[y]) == 0,
                 snt=ntype(x), dnt=ntype(y), nnx=float(d1), nny=float(d2), t=int(nodes[x]['t']),
                 xm=gx is not None, ym=gy is not None, x=x, y=y)
        if lab != 'NE':
            r['rl'] = run_len(x)
            if lab == 'FP':
                # FP subtype
                if gx is not None and gs.get(gx):
                    tgt = gs[gx]
                    r['fp'] = 'src_ok:' + ('tgt_unm' if gy is None else 'tgt_other')
                    # where is the true successor matched?
                    tt = [g2p.get(g) for g in tgt]
                    r['succ_pred'] = [ (None if q is None else [q, int(par.get(q, -1)) == x, par.get(q) is None]) for q in tt]
                    r['succ_dist'] = [float(np.linalg.norm(gpos[g] - pos[y])) for g in tgt]
                else:
                    r['fp'] = 'tgt_ok:' + ('src_unm' if gx is None else 'src_other')
            rows.append(r)
        elif lab == 'NE':
            pass
    # FN GT edges
    fns = []
    for u, v in GE:
        mu, mv = g2p.get(u), g2p.get(v)
        if mu is not None and mv is not None and mv in out.get(mu, []): continue
        if mu is None and mv is None: c = 'both_unm'
        elif mu is None: c = 'u_unm'
        elif mv is None: c = 'v_unm'
        elif out.get(mu) and mv in par: c = 'switch_both'
        elif out.get(mu): c = 'switch_out'
        elif mv in par: c = 'switch_in'
        else: c = 'break'
        rr = dict(m=name, set=st, c=c, t=int(gtt[u]))
        # nearest pred node distances to the unmatched GT node(s)
        for tag, g, mg in [('u', u, mu), ('v', v, mv)]:
            if mg is None:
                t = int(gtt[g]); tr, ns = trees.get(t, (None, None))
                if tr is not None:
                    dd, ii = tr.query(gpos[g], k=min(2, len(ns)))
                    dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
                    rr[tag + '_nd'] = float(dd[0]); n0 = ns[ii[0]]
                    rr[tag + '_nn_matched'] = n0 in p2g; rr[tag + '_nn_type'] = ntype(n0)
                    rr[tag + '_nn_deg'] = [n0 in par, len(out[n0])]
            else:
                rr[tag + '_type'] = ntype(mg); rr[tag + '_deg'] = [mg in par, len(out[mg])]
        if c == 'break':
            rr['gapd'] = float(np.linalg.norm(pos[mv] - pos[mu]))
        fns.append(rr)
    ne = Counter((r['lab'], r['et']) for r in rows)
    allc = Counter()
    for e in edges: allc[etype(e)] += 1
    return dict(m=name, set=st, rows=rows, fns=fns, allc=dict(allc), nnodes=len(nodes), ntype=dict(Counter(ntype(n) for n in nodes)))

if __name__ == '__main__':
    fs = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(60) as p: R = p.map(job, fs, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/an/p13_edges.json', 'w'))
    print('done', len(R))
