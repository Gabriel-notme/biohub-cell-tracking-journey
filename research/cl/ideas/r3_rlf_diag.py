"""Diagnostic for relinefit: how exactly can the B5 linefit input ('orig') be reconstructed from the pre-ILP fullgraph?
For each movie: reference nodes id-aligned with the fullgraph (same t), distance raw->ref, and whether linefit(orig) on the reference
structure reproduces the reference coordinates (all nodes, no subsampling), with and without the 3 um alignment cap.
Also: max |P14 - ref| for nodes kept from the reference graph, and P14 ids present in ref with a different t."""
import sys, json, glob, os
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
S = np.array([1.625, .40625, .40625]); W = 0.8
REF = {'hold36': '/workspace/runs/b5f_hold36/working/reference_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/reference_graphs',
       'audit32': '/workspace/sync3/runs/b5f_audit32/working/reference_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/reference_graphs',
       't127b': '/workspace/sync3/runs/b5f_t127b/working/reference_graphs'}
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def struct(edges, t_of):
    pred = defaultdict(list); succ = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in t_of and b in t_of and t_of[b] == t_of[a] + 1:
            succ[a].append(b); pred[b].append(a)
    return pred, succ


def hood(n, pred, succ, have):
    h = [(0, n)]; c = n
    for k in range(1, 3):
        p = pred.get(c, [])
        if len(p) != 1: break
        c = p[0]
        if c not in have: break
        h.append((-k, c))
    c = n
    for k in range(1, 3):
        s = succ.get(c, [])
        if len(s) != 1: break
        c = s[0]
        if c not in have: break
        h.append((k, c))
    return h


def lf(h, orig):
    dts = np.array([d for d, _ in h], float); X = np.stack([orig[m] for _, m in h])
    return np.array([np.polyval(np.polyfit(dts, X[:, a], 1), 0.0) for a in range(3)])


def job(a):
    s, f = a
    from edge_link import load_full
    name = os.path.basename(f)[:-5]
    R = json.load(open(f)); rn = {int(k): v for k, v in R['nodes'].items()}
    fids, fT, fV, fE, fprob = load_full(FULL[s] + '/' + name + '.geff')
    raw = {int(i): (int(t), v) for i, t, v in zip(fids.tolist(), fT.tolist(), fV)}
    t_ref = {k: int(v['t']) for k, v in rn.items()}
    pr, sr = struct(R['edges'], t_ref)
    out = defaultdict(int)
    orig_nc = {}; orig_c3 = {}; dists = []
    for k, v in rn.items():
        rp = np.array([v[c] for c in 'zyx'], float); fw = raw.get(k)
        if fw is not None and fw[0] == int(v['t']):
            d = float(np.linalg.norm((np.asarray(fw[1], float) - rp) * S)); dists.append(d)
            orig_nc[k] = np.asarray(fw[1], float); out['aligned'] += 1
            if d > 3.0: out['aligned_gt3um'] += 1
            orig_c3[k] = np.asarray(fw[1], float) if d <= 3.0 else rp
        else:
            orig_nc[k] = rp; orig_c3[k] = rp; out['unaligned'] += 1
            if fw is not None: out['id_in_full_other_t'] += 1
    for tag, orig in (('nocap', orig_nc), ('cap3', orig_c3)):
        for k in rn:
            h = hood(k, pr, sr, orig)
            rp = np.array([rn[k][c] for c in 'zyx'], float)
            pos = orig[k] if len(h) < 3 else (1 - W) * orig[k] + W * lf(h, orig)
            err = float(np.linalg.norm((pos - rp) * S))
            out[tag + '_ok' if err < 1e-3 else tag + '_bad'] += 1
            if err >= 1e-3 and all(m in raw for _, m in h): out[tag + '_bad_allaligned'] += 1
    P = json.load(open('/workspace/cl/ps_p14_%s/graphs/%s.json' % (s, name)))['nodes']
    mx = 0.0
    for k, v in P.items():
        k = int(k)
        if k in rn:
            if int(rn[k]['t']) != int(v['t']): out['p14_id_t_mismatch'] += 1; continue
            d = float(np.linalg.norm((np.array([v[c] for c in 'zyx']) - np.array([rn[k][c] for c in 'zyx'])) * S))
            mx = max(mx, d)
            if d > 1.5 + 1e-3: out['p14_minus_ref_gt1.5'] += 1
        else: out['p14_not_in_ref'] += 1
    out['p14_nodes'] = len(P); out['ref_nodes'] = len(rn)
    return name, s, dict(out), mx, (float(np.max(dists)) if dists else 0.0)


if __name__ == '__main__':
    jobs = []
    for s, d in REF.items():
        fs = sorted(glob.glob(d + '/*.json'))
        jobs += [(s, f) for f in fs[:int(sys.argv[1]) if len(sys.argv) > 1 else None]]
    with Pool(40) as p: res = p.map(job, jobs, chunksize=1)
    tot = defaultdict(int); mx = 0; md = 0
    for name, s, o, m, d in res:
        for k, v in o.items(): tot[k] += v
        mx = max(mx, m); md = max(md, d)
    print(len(res), 'movies'); print(json.dumps(dict(tot), indent=0)); print('max |p14-ref| um %.3f   max |raw-ref| aligned um %.3f' % (mx, md))
