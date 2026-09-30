"""r2_caps2 (diagnostic, reads GT): (a) P14 forks by weak-component size / min branch length / origin vs official label;
(b) div_complete candidate ring: candidates on the B5 graph with max_pb 16 vs the 13 um cap, GT-positive counts per type.
Writes /workspace/cl/ideas/r2_caps2_out.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
os.environ['BIOHUB_ART'] = '/workspace/models/b56/artifact_bundle'
for p in ['/workspace/models/b56/artifact_bundle', '/workspace/cl', '/workspace/cl/ideas', '/workspace/official/src', '/workspace/code', '/workspace/p56stage']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def matching(nodes, edges, gt):
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    return pred, inv, {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    import evalx, combo14, div_complete as dc
    from tracking_cellmot.division_metrics import score_divisions
    K = evalx.K
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]; emb = name[:4]
    gt, _ = evalx.load_gt(name)
    ea = gt.edge_attrs()
    gout = defaultdict(list)
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gout[int(x)].append(int(y))
    gdiv = {g: set(c) for g, c in gout.items() if len(c) >= 2}
    # (a) P14 forks
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, fullgeff=FULL[st] + '/' + name + '.geff')
    out = defaultdict(list); par = {}; eflag = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
        eflag[(a, b)] = 'dc' if 'div_complete' in e else ('el' if 'edge_link' in e else ('rl' if 'relink' in e else ('ll' if 'long_link' in e else 'b5')))
    adj = defaultdict(list)
    for a, bs in out.items():
        for b in bs: adj[a].append(b); adj[b].append(a)
    comp = {}; csize = {}
    for n in nodes:
        if n in comp: continue
        st_ = [n]; comp[n] = n; mem = [n]
        while st_:
            x = st_.pop()
            for y in adj.get(x, []):
                if y not in comp: comp[y] = n; st_.append(y); mem.append(y)
        csize[n] = len(mem)
    pred, inv, p2g = matching(nodes, edges, gt)
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tpf = {inv[int(x)] for x in res.tp_forks}; fpf = {inv[int(x)] for x in res.fp_forks}

    def blen(x, lim=60):
        L = 1
        while len(out.get(x, [])) == 1 and L < lim: x = out[x][0]; L += 1
        return L, len(out.get(x, []))

    def hlen(x, lim=60):
        h = 0
        while x in par and len(out[par[x]]) == 1 and h < lim: x = par[x]; h += 1
        return h
    forks = []
    for n in nodes:
        k = out.get(n, [])
        if len(k) < 2: continue
        bl = [blen(c) for c in k]
        pos = lambda m: np.array([nodes[m][q] for q in 'zyx'], float) * S
        forks.append(dict(m=name, emb=emb, lab='tp' if n in tpf else ('fp' if n in fpf else 'nc'), cs=csize[comp[n]], bmin=min(b[0] for b in bl),
                          bmax=max(b[0] for b in bl), bend=[b[1] for b in bl], hist=hlen(n), t=int(nodes[n]['t']), org=sorted(eflag[(n, c)] for c in k),
                          dab=float(np.linalg.norm(pos(k[0]) - pos(k[1]))), dpk=max(float(np.linalg.norm(pos(c) - pos(n))) for c in k),
                          pm=n in p2g, pdiv=(p2g.get(n) in gdiv) if n in p2g else False))
    # (b) div_complete ring on the B5 graph
    bn, be = evalx.load_graph_json(B5[st] + '/' + name + '.json')
    rows, bout, bprev, bpos = dc.candidates(bn, be, max_pb=16.0, max_ab=20.0)
    _, _, b2g = matching(bn, be, gt)
    ring = Counter()
    for p, a, b, q, typ in rows:
        dpb = float(np.linalg.norm(bpos[b] - bpos[p]))
        gp = b2g.get(p)
        posl = gp in gdiv and b2g.get(a) in gdiv[gp] and b2g.get(b) in gdiv[gp]
        ring[(emb, typ, 'in13' if dpb <= 13 else 'ring', 'P' if posl else 'N')] += 1
    return dict(m=name, forks=forks, ring=[list(k) + [v] for k, v in ring.items()])


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    if len(sys.argv) > 1: files = files[:int(sys.argv[1])]
    with Pool(40, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r2_caps2_out.json', 'w'))
    ring = Counter()
    for r in R:
        for k in r['ring']: ring[tuple(k[:-1])] += k[-1]
    for k in sorted(ring): print('ring', k, ring[k])
    F = [x for r in R for x in r['forks']]
    c = Counter()
    for x in F:
        c[(x['emb'], x['lab'], 'cs<6' if x['cs'] < 6 else ('cs<12' if x['cs'] < 12 else 'cs>=12'))] += 1
    for k in sorted(c): print('fork cs', k, c[k])
    c = Counter()
    for x in F:
        c[(x['emb'], x['lab'], 'bmin%d' % min(x['bmin'], 5))] += 1
    for k in sorted(c): print('fork bmin', k, c[k])
