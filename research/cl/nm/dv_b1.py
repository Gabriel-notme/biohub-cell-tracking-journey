"""dv_b1: b1 fork probability + node embeddings for every start-type row of dv_cand (real candidates and synthetic fork births).
phase 'patch' (CPU pool): patches + fork geometry per movie -> /workspace/cl/nm/dv_patch/<set>__<movie>.npz
phase 'gpu' <gpu>: encode with b1, fork logits -> /workspace/cl/nm/dv_b1_feats.json ({movie|p|a|b: {b1_fork: ..}}) and dv_b1_emb.npz"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
for p in ['/workspace/cl', '/workspace/code', ART, '/workspace/p12ds']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
PD = Path('/workspace/cl/nm/dv_patch')


def job(f):
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    from cell_event import SCALE, Movie, chain, fork_geometry
    s, name = os.path.basename(f)[:-5].split('__')
    of = PD / ('%s__%s.npz' % (s, name))
    if of.exists(): return 1
    R = [r for r in json.load(open(f)) if r['typ'] == 0 and (r['src'] == 'fork' or r['lab'] != 'U')]
    if not R: np.savez(of, n=0); return 1
    d = json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, name)))
    nodes = {int(k): v for k, v in d['nodes'].items()}
    out, prev = defaultdict(list), {}
    for e in d['edges']:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    need = sorted({x for r in R for x in (r['p'], r['a'], r['b'])}, key=lambda n: (nodes[n]['t'], n))
    idx = {n: i for i, n in enumerate(need)}
    mv = Movie(Path('/workspace/data/train') / (name + '.zarr'), context=5)
    X = mv.patches([nodes[n]['t'] for n in need], [[nodes[n][k] for k in 'zyx'] for n in need])
    G, T, K = [], [], []
    for r in R:
        p, a, b = r['p'], r['a'], r['b']
        # p history: for a synthetic birth p is a fork; its history chain is unchanged. a/b forward chains unchanged.
        o2 = out
        if r['src'] == 'fork':
            o2 = out.copy(); o2[p] = [a]
        G.append(fork_geometry(chain(p, prev, pos), chain(a, o2, pos), chain(b, o2, pos)))
        T.append([idx[p], idx[a], idx[b]]); K.append('%s|%d|%d|%d' % (name, p, a, b))
    np.savez(of, n=len(R), X=X, G=np.stack(G).astype(np.float32), T=np.array(T), K=np.array(K))
    return 1


def gpu(dev):
    os.environ['CUDA_VISIBLE_DEVICES'] = dev
    import torch
    from cell_event import load_event_model
    model, _ = load_event_model(Path(ART) / 'b1_best.pt', 'cuda')
    feats = {}; EK, EP, EA, EB = [], [], [], []
    for f in sorted(PD.glob('*.npz')):
        z = np.load(f)
        if int(z['n']) == 0: continue
        X = torch.from_numpy(z['X']).cuda().float()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            E = torch.cat([model.encode(X[i:i + 512]).float() for i in range(0, len(X), 512)])
            T = torch.from_numpy(z['T']).cuda()
            lg = model.fork_logits(E[T[:, 0]], E[T[:, 1]], E[T[:, 2]], torch.from_numpy(z['G']).cuda()).float()
            ph = model.phase(E).float().squeeze(-1)
        lg = lg.cpu().numpy(); ph = ph.cpu().numpy(); T = z['T']; E = E.cpu().numpy()
        for k, l, t in zip(z['K'], lg, T):
            feats[str(k)] = dict(b1_fork=float(1 / (1 + np.exp(-l))), b1_logit=float(l), b1_ph_p=float(ph[t[0]]), b1_ph_a=float(ph[t[1]]), b1_ph_b=float(ph[t[2]]))
        EK += [str(k) for k in z['K']]; EP.append(E[T[:, 0]]); EA.append(E[T[:, 1]]); EB.append(E[T[:, 2]])
    json.dump(feats, open('/workspace/cl/nm/dv_b1_feats.json', 'w'))
    np.savez('/workspace/cl/nm/dv_b1_emb.npz', K=np.array(EK), P=np.concatenate(EP), A=np.concatenate(EA), B=np.concatenate(EB))
    print('gpu done', len(feats))


if __name__ == '__main__':
    if sys.argv[1] == 'patch':
        PD.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob('/workspace/cl/nm/dv_cand/*.json'))
        with Pool(24, maxtasksperchild=2) as p: print('patch done', sum(p.map(job, files, chunksize=1)))
    else:
        gpu(sys.argv[2])
