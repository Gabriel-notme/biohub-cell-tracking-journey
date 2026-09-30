"""Why are real (non-flip) link fixes absent from the pre-ILP candidate graph? Node provenance, distances, current-edge probs."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLOSC_NTHREADS']: os.environ[_k] = '1'
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/code')
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
S = np.array([1.625, .40625, .40625])


def job(f):
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import edge_link
    d = json.load(open(f)); s, m = d['set'], d['movie']
    g = json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, m)))
    nodes = g['nodes']; ed = {(int(e['source_id']), int(e['target_id'])): e for e in g['edges']}
    fids, fT, fV, fE, fprob = edge_link.load_full(FULL[s] + '/' + m + '.geff')
    fset = set(fids.tolist()); fpos = {int(i): v * S for i, v in zip(fids.tolist(), fV)}
    fout = defaultdict(list)
    for (a, b), p in zip(fE.tolist(), fprob.tolist()): fout[int(a)].append((int(b), float(p)))
    out = []
    for x in d['fixes']:
        if x['typ'].startswith('div_') or x['flip'] or 'TP' in x['rml']: continue
        p1, p2 = x['p1'], x['p2']
        n1, n2 = nodes[str(p1)], nodes[str(p2)]
        tag = lambda n: next((k for k in ('gap_synthetic', 'tb_ext', 'learned_recovery', 'edge_link_node') if k in n), 'det' if int(n['node_id']) in fset else 'other')
        # has p1 any pre-ILP out edges? how many; rank of p2 among p1's pre-ILP neighbours by distance
        no = len(fout.get(p1, []))
        moved = [float(np.linalg.norm(np.array([n[k] for k in 'zyx'], float) * S - fpos[int(n['node_id'])])) if int(n['node_id']) in fpos else -1. for n in (n1, n2)]
        rmp = [ed[tuple(r)].get('edge_prob', -1) if tuple(r) in ed else -1 for r in x['rm']]
        out.append(dict(emb=m[:4], set=s, typ=x['typ'], pre=int(x['fe'] >= 0), tag1=tag(n1), tag2=tag(n2), no=no, dist=x['dist'], moved=moved, rmprov=x['rmprov'], rmp=rmp, dpg=x['dpg'], dcg=x['dcg'], dqg=x['dqg']))
    return out


if __name__ == '__main__':
    R = [r for rs in Pool(20).map(job, sorted(glob.glob('/workspace/cl/nm/rl_oracle/*.json'))) for r in rs]
    json.dump(R, open('/workspace/cl/nm/rl_nopre_rows.json', 'w'))
    for emb in ['44b6', '6bba']:
        rr = [r for r in R if r['emb'] == emb]
        print('==', emb, len(rr))
        print(' tags', Counter((r['pre'], r['tag1'], r['tag2']) for r in rr).most_common(12))
        print(' typ', Counter((r['pre'], r['typ']) for r in rr).most_common())
        print(' p1 has pre-ILP out edges', Counter((r['pre'], r['no'] > 0) for r in rr))
        dd = np.array([r['dist'] for r in rr]); print(' dist quantiles', np.round(np.quantile(dd, [.1, .25, .5, .75, .9]), 1))
        print(' rm prov', Counter(tuple(r['rmprov']) for r in rr).most_common(8))
        rp = [p for r in rr for p in r['rmp'] if p is not None and p >= 0]; print(' rm edge prob quantiles', np.round(np.quantile(rp, [.1, .25, .5, .75, .9]), 3) if rp else None)
        mv = [x for r in rr for x in r['moved'] if x >= 0]; print(' node moved vs pre-ILP (um) quantiles', np.round(np.quantile(mv, [.5, .9, .99]), 2))
