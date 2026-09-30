"""Localization headroom on P15 (GT-oracle). Per movie:
 - residuals of matched (pred, GT) pairs (rounded pred coords, um)
 - 'intended' GT node of each pred node from track consistency (votes: own match w=1, pred-chain neighbours at k<=3 steps
   whose matched GT node is walked k steps through GT, weight 1/k); one pred node per GT node (highest vote, then closest)
 - FN edge causes (unmatched endpoint with nearest pred at 0-7 / 7-9 / 9-12 / >12 um, flip = both matched but pred track
   continues on another node that intends the GT node, link = rest)
 - oracle variants: move intended pred nodes (d <= R) a fraction a toward their GT node; rescore with the official metric.
usage: loc_oracle.py <out.json>"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
# (label, R_um, alpha, mode) mode: all | matched_same (only nodes whose intended == current match) | unmatched (only currently unmatched) | flip (only matched nodes whose intended != match)
VARIANTS = [('snap_matched_same', 99, 1.0, 'matched_same'), ('R5', 5, 1.0, 'all'), ('R7', 7, 1.0, 'all'), ('R9', 9, 1.0, 'all'),
            ('R12', 12, 1.0, 'all'), ('R20', 20, 1.0, 'all'), ('R12_a0.5', 12, 0.5, 'all'), ('R12_a0.3', 12, 0.3, 'all'),
            ('R12_unmatched', 12, 1.0, 'unmatched'), ('R12_flip', 12, 1.0, 'flip'), ('R7_unmatched_bin5_7', 7, 1.0, 'unmatched')]


def rpos(v):
    return np.array([max(0, int(round(v[k]))) for k in 'zyx'], float) * S


def job(args):
    s, f = args
    import evalx
    from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
    K = evalx.K
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    er = evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    base = per_sample_metrics(er, n_total, node_recall(pred, gt)); base['movie'] = name; base['n_total'] = n_total; base['set'] = s; base['vi'] = 0
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x], float) * S for i, z, y, x in zip(*[ga[k].to_list() for k in [K.NODE_ID, 'z', 'y', 'x']])}
    graw = {int(i): np.array([z, y, x], float) for i, z, y, x in zip(*[ga[k].to_list() for k in [K.NODE_ID, 'z', 'y', 'x']])}
    gt_t = {int(i): int(t) for i, t in zip(ga[K.NODE_ID].to_list(), ga['t'].to_list())}
    ea = gt.edge_attrs(); GE = [(int(a), int(b)) for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list())]
    gsucc = defaultdict(list); gpar = {}
    for a, b in GE: gsucc[a].append(b); gpar[b] = a
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    pos = {n: rpos(v) for n, v in nodes.items()}
    byt = defaultdict(list)
    for n, v in nodes.items(): byt[int(v['t'])].append(n)
    from scipy.spatial import cKDTree
    trees = {t: (cKDTree(np.stack([pos[n] for n in ns])), ns) for t, ns in byt.items()}

    # ---- intended GT per pred node (track-consistency votes)
    def gwalk_fwd(g, chain):  # chain: pred nodes at t+1..t+k (last = target); pick GT child closest to pred chain node at divisions
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
                chain = list(reversed(back[:-1])) + [n]  # pred nodes from t(p)+1 .. t(n)
                g = gwalk_fwd(p2g[p], chain)
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
    intended = {n: votes[n] for n in claim.values()}

    # ---- residuals of matched pairs
    resid = [float(np.linalg.norm(pos[p] - gpos[g])) for p, g in p2g.items()]
    hist_edges = [0, 1, 2, 3, 4, 5, 6, 7.0001]
    rh = np.histogram(resid, bins=hist_edges)[0].tolist()
    # distance of intended-but-not-matched-to-it nodes
    cat_int = Counter()
    for n, (g, w, d) in intended.items():
        m = p2g.get(n)
        if m == g: cat_int['same'] += 1
        elif m is None: cat_int['unm_%s' % ('lt7' if d <= 7 else '7_9' if d <= 9 else '9_12' if d <= 12 else 'gt12')] += 1
        else: cat_int['flip_%s' % ('lt5' if d <= 5 else '5_7' if d <= 7 else '7_9' if d <= 9 else 'gt9')] += 1

    # ---- FN edge causes
    gi = {g: n for n, (g, w, d) in intended.items()}
    fn = Counter()
    for u, v in GE:
        mu, mv = g2p.get(u), g2p.get(v)
        if mu is not None and mv is not None and mv in succ.get(mu, []): fn['tp'] += 1; continue
        if mu is None or mv is None:
            dd = []
            for g in [u, v]:
                if g in g2p: continue
                t = gt_t[g]
                if t in trees:
                    tr, ns = trees[t]; q, j = tr.query(gpos[g]); dd.append(q)
                else: dd.append(99.)
            dmax = max(dd)
            b = 'lt7' if dmax <= 7 else '7_9' if dmax <= 9 else '9_12' if dmax <= 12 else 'gt12'
            has_int = all((g in g2p) or (g in gi) for g in [u, v])
            fn['miss_%s_%s' % (b, 'int' if has_int else 'noint')] += 1
        else:
            x = [c for c in succ.get(mu, []) if gi.get(v) == c]
            y = gi.get(u) == par.get(mv)
            if x or y: fn['flip'] += 1
            else: fn['link'] += 1

    # ---- oracle variants
    rows = [base]
    for vi, (lab, R, al, mode) in enumerate(VARIANTS, 1):
        nn = {n: dict(v) for n, v in nodes.items()}; moved = 0
        for n, (g, w, d) in intended.items():
            m = p2g.get(n)
            if mode == 'matched_same' and m != g: continue
            if mode == 'unmatched' and m is not None: continue
            if mode == 'flip' and (m is None or m == g): continue
            if d > R: continue
            if lab == 'R7_unmatched_bin5_7' and d < 5: continue
            cur = np.array([nodes[n][k] for k in 'zyx'], float)
            new = cur + al * (graw[g] - cur)
            nn[n].update(dict(zip('zyx', map(float, new)))); moved += 1
        r = evalx.score_movie(name, nn, edges); r['set'] = s; r['vi'] = vi; r['moved'] = moved; rows.append(r)
    info = dict(movie=name, set=s, resid_hist=rh, resid_mean=float(np.mean(resid)) if resid else 0., n_matched=len(p2g), n_pred=len(nodes),
                n_gt=len(gpos), intended=dict(cat_int), fn=dict(fn), n_intended=len(intended))
    return rows, info


if __name__ == '__main__':
    fs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p15_%s/graphs/*.json' % s))]
    with Pool(24, maxtasksperchild=4) as p: out = p.map(job, fs, chunksize=1)
    rows = [r for rs, _ in out for r in rs]; infos = [i for _, i in out]
    json.dump(rows, open(sys.argv[1], 'w')); json.dump(infos, open(sys.argv[1].replace('.json', '_info.json'), 'w'))
    from tracking_cellmot.metrics import summarise
    base = {r['movie']: r for r in rows if r['vi'] == 0}; ms = sorted(base)
    for vi, (lab, R, al, mode) in enumerate(VARIANTS, 1):
        cur = {r['movie']: r for r in rows if r['vi'] == vi}
        line = '%-22s' % lab
        for emb in ['44b6', '6bba', '']:
            mm = [m for m in ms if m.startswith(emb)]
            a = summarise([base[m] for m in mm]); b = summarise([cur[m] for m in mm])
            line += ' | %s %+.5f (tp %+d fp %+d fn %+d)' % (emb or 'all', b['score'] - a['score'], sum(cur[m]['edge_tp'] - base[m]['edge_tp'] for m in mm),
                                                            sum(cur[m]['edge_fp'] - base[m]['edge_fp'] for m in mm), sum(cur[m]['edge_fn'] - base[m]['edge_fn'] for m in mm))
        line += ' | moved %d' % sum(cur[m]['moved'] for m in ms)
        print(line, flush=True)
    for emb in ['44b6', '6bba']:
        I = [i for i in infos if i['movie'].startswith(emb)]
        rh = np.sum([i['resid_hist'] for i in I], 0); print(emb, 'matched', sum(i['n_matched'] for i in I), 'resid hist 0-1..6-7um', rh.tolist(),
                                                          'mean %.3f' % (sum(i['resid_mean'] * i['n_matched'] for i in I) / max(1, sum(i['n_matched'] for i in I))))
        c = Counter();
        for i in I: c.update(i['intended'])
        print('  intended', dict(sorted(c.items())))
        c = Counter()
        for i in I: c.update(i['fn'])
        print('  gt edges', dict(sorted(c.items())))
