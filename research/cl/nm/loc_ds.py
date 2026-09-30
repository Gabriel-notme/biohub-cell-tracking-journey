"""LOEO regression dataset: pred nodes of P15 with a track-consistent intended GT node (loc_oracle definition), patch at the node's
current float position (B5 Movie 7x12x24x24), target = GT - pred (um, no cutoff up to 12 um).
typ 0 = matched to intended, d<=5 | 1 = matched, d>5 (tail) | 2 = unmatched, intended d<=12 (U, the oracle movers).
44b6: all nodes; 6bba: all typ 1/2 + 35% of typ 0.  -> /workspace/cl/nm/loc_ds/<movie>.npz"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/art_b56/artifact_bundle')
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
OUT = '/workspace/cl/nm/loc_ds'


def rpos(v):
    return np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S


def intended_map(nodes, edges, name):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    graw = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(*[ga[k].to_list() for k in [K.NODE_ID, 'z', 'y', 'x']])}
    gpos = {g: v * S for g, v in graw.items()}
    gt_t = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs()
    gsucc = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(a)].append(int(b)); gpar[int(b)] = int(a)
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: rpos(v) for n, v in nodes.items()}

    def gwalk_fwd(g, chain):
        for q in chain:
            ch = gsucc.get(g, [])
            if not ch: return None
            g = ch[0] if len(ch) == 1 else min(ch, key=lambda c: np.linalg.norm(gpos[c] - pos[q]))
        return g

    def gwalk_back(g, k):
        for _ in range(k):
            g = gpar.get(g)
            if g is None: return None
        return g
    votes = {}
    for n in nodes:
        v = Counter()
        if n in p2g: v[p2g[n]] += 1.0
        c = n; back = []
        for k in range(1, 4):
            p = par.get(c)
            if p is None: break
            back.append(p); c = p
            if p in p2g:
                g = gwalk_fwd(p2g[p], list(reversed(back[:-1])) + [n])
                if g is not None: v[g] += 1.0 / k
        c = n
        for k in range(1, 4):
            ch = succ.get(c, [])
            if len(ch) != 1: break
            c = ch[0]
            if c in p2g:
                g = gwalk_back(p2g[c], k)
                if g is not None: v[g] += 1.0 / k
        if v:
            g, w = max(v.items(), key=lambda kv: (kv[1], -np.linalg.norm(gpos[kv[0]] - pos[n])))
            if w >= 1.0 and gt_t.get(g) == int(nodes[n]['t']): votes[n] = (g, w, float(np.linalg.norm(gpos[g] - pos[n])))
    claim = {}
    for n, (g, w, d) in votes.items():
        if g not in claim or (w, -d) > (votes[claim[g]][1], -votes[claim[g]][2]): claim[g] = n
    return {n: votes[n] for n in claim.values()}, p2g, graw


def job(args):
    s, f = args
    import numcodecs.blosc; numcodecs.blosc.use_threads = False
    import evalx
    from cell_event import Movie, STRIDE, SCALE
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    inten, p2g, graw = intended_map(nodes, edges, name)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    ids, typ = [], []
    for n, (g, w, d) in inten.items():
        m = p2g.get(n)
        if m is None:
            if d <= 12: ids.append(n); typ.append(2)
            continue
        if m != g: continue
        ty = 1 if d > 5 else 0
        if ty == 0 and name.startswith('6bba') and rng.random() > 0.35: continue
        ids.append(n); typ.append(ty)
    if not ids: return name, 0
    order = sorted(range(len(ids)), key=lambda i: nodes[ids[i]]['t']); ids = [ids[i] for i in order]; typ = [typ[i] for i in order]
    coords = np.array([[nodes[n][k] for k in 'zyx'] for n in ids], float); times = np.array([int(nodes[n]['t']) for n in ids])
    tgt = np.array([(graw[inten[n][0]] - coords[i]) * S for i, n in enumerate(ids)], np.float32)
    frac = ((coords - np.rint(coords / STRIDE) * STRIDE) * SCALE).astype(np.float32)
    x = Movie('/workspace/data/train/%s.zarr' % name, 7).patches(times, coords)
    np.savez('%s/%s.npz' % (OUT, name), x=x, y=tgt, f=frac, typ=np.array(typ, np.int8), ids=np.array(ids), t=times, c=coords.astype(np.float32),
             d0=np.array([inten[n][2] for n in ids], np.float32))
    return name, len(ids)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(24, maxtasksperchild=4) as p: out = p.map(job, fs, chunksize=1)
    print('movies', len(out), 'nodes', sum(n for _, n in out))
