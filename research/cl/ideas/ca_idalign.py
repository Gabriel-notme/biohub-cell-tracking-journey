"""Audit: are B5 lineage-graph node IDs the same detections as the pre-ILP full-graph IDs? (per set)"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/workspace/p56stage')
S = np.array([1.625, .40625, .40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(a):
    s, f = a
    import edge_link
    name = os.path.basename(f)[:-5]
    d = json.load(open(f)); nodes = {int(k): v for k, v in d['nodes'].items()}
    fids, fT, fV, fE, fprob = edge_link.load_full(FULL[s] + '/' + name + '.geff')
    idx = {int(i): j for j, i in enumerate(fids.tolist())}
    common = [n for n in nodes if n in idx]
    dev = np.array([np.linalg.norm((np.array([nodes[n][k] for k in 'zyx'], float) - fV[idx[n]]) * S) for n in common])
    tdev = np.array([int(nodes[n]['t']) != int(fT[idx[n]]) for n in common])
    bad = (dev > 5.0) | tdev
    # edge attribute types of B5 edges touching non-full or bad nodes
    badset = {n for n, b in zip(common, bad) if b} | {n for n in nodes if n not in idx}
    ek = Counter()
    for e in d['edges']:
        sid, tid = int(e['source_id']), int(e['target_id'])
        if sid in badset or tid in badset:
            ek[','.join(sorted(k for k in e if k not in ('source_id', 'target_id', 'distance_um', 'edge_prob')))] += 1
    # max id of B5 nodes vs full ids
    return dict(set=s, movie=name, n=len(nodes), nfull=len(fids), common=len(common), notfull=len(nodes) - len(common),
                bad=int(bad.sum()), tdev=int(tdev.sum()), dev_q=[float(np.quantile(dev, q)) for q in (.5, .99, .999)] if len(dev) else [],
                maxid_b5=max(nodes), maxid_full=int(fids.max()), ek=dict(ek))


if __name__ == '__main__':
    jobs = [(s, f) for s in B5 for f in sorted(glob.glob(B5[s] + '/*.json'))]
    with Pool(48) as p: R = p.map(job, jobs, chunksize=1)
    for s in B5:
        rr = [r for r in R if r['set'] == s]
        ek = Counter()
        for r in rr: ek.update(r['ek'])
        print(s, 'movies', len(rr), 'nodes', sum(r['n'] for r in rr), 'notfull', sum(r['notfull'] for r in rr), 'bad(common, >5um or t mismatch)', sum(r['bad'] for r in rr),
              'tdev', sum(r['tdev'] for r in rr), 'median dev q50/q99/q999', np.round(np.median([r['dev_q'] for r in rr], 0), 3).tolist(),
              'movies with bad>0', sum(r['bad'] > 0 for r in rr), 'maxid b5>full', sum(r['maxid_b5'] > r['maxid_full'] for r in rr))
        print('   edge kinds touching bad/notfull nodes', ek.most_common(8))
    json.dump(R, open('/workspace/cl/ideas/ca_idalign.json', 'w'))
