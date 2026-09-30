"""READ-ONLY: per-frame census of P14 track starts / ends (>= minlen nodes) and node counts, plus pre-ILP detection counts
(kept vs dropped) per frame; shows whether the temporal boundary frames lose detections/links."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/p56stage')
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def job(f):
    try:
        import numcodecs.blosc
        numcodecs.blosc.use_threads = False
    except Exception:
        pass
    from edge_link import load_full
    st = [s for s in SETS if '_%s/' % s in f][0]; name = f.split('/')[-1][:-5]
    d = json.load(open(f)); nodes = {int(k): v for k, v in d['nodes'].items()}
    par = {}; succ = defaultdict(list)
    for e in d['edges']: par[int(e['target_id'])] = int(e['source_id']); succ[int(e['source_id'])].append(int(e['target_id']))

    def clen(n, down, cap=5):
        L = 1; c = n
        while L < cap:
            nx = succ.get(c, []) if down else ([par[c]] if c in par else [])
            if len(nx) != 1: break
            c = nx[0]; L += 1
        return L
    starts = Counter(); ends = Counter(); nt = Counter()
    for n, v in nodes.items():
        t = int(v['t']); nt[t] += 1
        if n not in par and clen(n, True) >= 5: starts[t] += 1
        if not succ.get(n) and clen(n, False) >= 5: ends[t] += 1
    fids, fT, fV, fE, fprob = load_full(FULL[st] + '/' + name + '.geff')
    kept = Counter(); drop = Counter()
    for i, t in zip(fids.tolist(), fT.tolist()):
        (kept if int(i) in nodes else drop)[int(t)] += 1
    return dict(emb=name[:4], starts=starts, ends=ends, nt=nt, kept=kept, drop=drop)


if __name__ == '__main__':
    files = [f for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))]
    with Pool(24) as p: R = p.map(job, files, chunksize=1)
    for emb in ['44b6', '6bba']:
        rr = [r for r in R if r['emb'] == emb]
        agg = {k: Counter() for k in ['starts', 'ends', 'nt', 'kept', 'drop']}
        for r in rr:
            for k in agg: agg[k].update(r[k])
        print('==', emb, 'movies', len(rr))
        print(' t  | nodes  starts(>=5) ends(>=5) | preILP kept dropped drop_frac')
        for t in list(range(0, 8)) + [20, 50, 80] + list(range(92, 100)):
            print('%3d | %6d  %6d  %6d | %6d %6d %.3f' % (t, agg['nt'][t], agg['starts'][t], agg['ends'][t], agg['kept'][t], agg['drop'][t],
                                                     agg['drop'][t] / max(1, agg['kept'][t] + agg['drop'][t])))
