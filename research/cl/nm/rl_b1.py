"""b1 image features for rl_cands rows: edge-head probabilities of the proposed edge and of the edges it competes with,
plus embedding cosine similarities.  usage: rl_b1.py <gpu> <k> <n>  (process every n-th movie starting at k)
-> /workspace/cl/nm/rl_b1f/<set>__<movie>.pkl  {X, names}"""
import os, sys, json, glob, pickle, time
gpu, K, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLOSC_NTHREADS']: os.environ[_k] = '1'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/code')
import numcodecs.blosc
numcodecs.blosc.use_threads = False
from pathlib import Path
from collections import defaultdict
import numpy as np
OUT = Path('/workspace/cl/nm/rl_b1f'); OUT.mkdir(exist_ok=True, parents=True)
NAMES = ['b_sd', 'b_sc', 'b_qd', 'b_qc', 'b_dsc', 'b_dqd', 'b_swap', 'cos_sd', 'cos_sc', 'cos_qd', 'cos_dc']


def main():
    import torch
    torch.set_num_threads(2)
    from refine_events import EventRefiner, batch_edge_geometry
    er = EventRefiner([Path(ART) / 'b1_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, '/workspace/cl/nm/rl_b1cache_%s' % gpu)
    files = sorted(glob.glob('/workspace/cl/nm/rl_cands/*.pkl'))[K::N]
    for f in files:
        s_, name = os.path.basename(f)[:-4].split('__'); of = OUT / os.path.basename(f)
        if of.exists(): continue
        t0 = time.time()
        z = pickle.load(open(f, 'rb')); L = z['L']
        g = json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s_, name)))
        nodes = {int(k): v for k, v in g['nodes'].items()}
        out, prev = defaultdict(list), {}
        for e in g['edges']:
            a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
        pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * np.array([1.625, .40625, .40625], np.float32) for n, v in nodes.items()}
        pairs = set()
        for s, d, c, q in zip(L['s'], L['d'], L['c'], L['q']):
            pairs.add((int(s), int(d)))
            if c >= 0: pairs.add((int(s), int(c)))
            if q >= 0: pairs.add((int(q), int(d)))
            if c >= 0 and q >= 0: pairs.add((int(q), int(c)))
        pairs = sorted(pairs)
        need = {x for pr in pairs for x in pr}
        if not pairs:
            pickle.dump(dict(X=np.zeros((0, len(NAMES)), np.float32), names=NAMES), open(of, 'wb')); continue
        ids, emb = er.embeddings(name, {n: nodes[n] for n in need}); lk = {n: i for i, n in enumerate(ids)}
        geo = batch_edge_geometry(pairs, prev, out, pos)
        pr = er.score('edge', pairs, geo, emb, lk); Pm = dict(zip(pairs, map(float, pr)))
        E = emb[0]; En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
        cs = lambda a, b: float(En[lk[a]] @ En[lk[b]])
        X = []
        for s, d, c, q in zip(L['s'], L['d'], L['c'], L['q']):
            s, d, c, q = int(s), int(d), int(c), int(q)
            bsd = Pm[(s, d)]; bsc = Pm[(s, c)] if c >= 0 else -1.; bqd = Pm[(q, d)] if q >= 0 else -1.; bqc = Pm[(q, c)] if (c >= 0 and q >= 0) else -1.
            X.append([bsd, bsc, bqd, bqc, bsd - bsc if c >= 0 else 1., bsd - bqd if q >= 0 else 1.,
                      (bsd + bqc - bsc - bqd) if (c >= 0 and q >= 0) else -9., cs(s, d), cs(s, c) if c >= 0 else -2., cs(q, d) if q >= 0 else -2., cs(d, c) if c >= 0 else -2.])
        pickle.dump(dict(X=np.array(X, np.float32).reshape(-1, len(NAMES)), names=NAMES), open(of, 'wb'))
        print('B1F', name, len(pairs), len(need), round(time.time() - t0, 1), flush=True)
        for p in Path('/workspace/cl/nm/rl_b1cache_%s' % gpu).glob(name + '*'): p.unlink()


if __name__ == '__main__':
    main()
