"""ver_b1: b1 image features for every ver_cands candidate (CPU, one movie per worker, torch 1 thread).
Per candidate key: [b1 edge prob of the new link, cosine(emb src, emb dst), cosine(emb terminus, emb new node),
 nucleus mean intensity of new node / terminus, nucleus std ratio, b1 edge prob of the terminus' own last track edge (reference)]
Writes /workspace/cl/nm/ver_b1feat.json {movie: {key: [..]}}"""
import os, sys, json, glob, time
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ART = '/workspace/art_b56/artifact_bundle'
sys.path.insert(0, ART); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625], np.float32)
NAMES = ['b1p', 'cos_sd', 'cos_term', 'int_ratio', 'std_ratio', 'b1p_ref']


def job(f):
    import numcodecs.blosc
    numcodecs.blosc.use_threads = False
    import torch
    torch.set_num_threads(1)
    from cell_event import Movie, chain, edge_geometry, load_event_model
    t0 = time.time()
    d = json.load(open(f)); name = d['movie']; st = d['set']
    G = json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json' % (st, name)))
    nodes = {int(k): v for k, v in G['nodes'].items()}
    out = defaultdict(list); prev = {}
    for e in G['edges']:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * S for n, v in nodes.items()}
    model, _ = load_event_model(ART + '/b1_best.pt', 'cpu'); model.float()
    need = {}  # node key -> (t, zyx px)

    def reg(k, t, zyx):
        need[k] = (int(t), [float(c) for c in zyx])
    items = []  # (key, src_key, dst_key, hist positions, fut positions, terminus key, new key)
    for w in d['walks']:
        n, fwd = int(w['seed']), bool(w['fwd'])
        reg(('p', n), nodes[n]['t'], [nodes[n][c] for c in 'zyx'])
        tr = chain(n, prev, pos, 4) if fwd else chain(n, out, pos, 4)  # terminus track (backward for END, forward for START)
        # reference: terminus' own last track edge
        spos = [pos[n]]
        steps = w['steps']
        for k, s in enumerate(steps):
            reg(('d', int(s['x'])), s['t'], [s['z'], s['y'], s['xx']])
        wpos = [np.array([s['z'], s['y'], s['xx']], np.float32) * S for s in steps]
        for k, s in enumerate(steps):
            cur_key = ('p', n) if k == 0 else ('d', int(steps[k - 1]['x']))
            new_key = ('d', int(s['x']))
            back = ([wpos[k - 1 - i] for i in range(k)] + list(tr))[:4]  # positions from cur going away from the walk direction
            ahead = wpos[k:k + 4]
            if fwd: items.append(('%d_%d_s%d' % (n, 1, k), cur_key, new_key, back, ahead, ('p', n), new_key))
            else: items.append(('%d_%d_s%d' % (n, 0, k), new_key, cur_key, ahead, back, ('p', n), new_key))
        if w['join']:
            y = int(w['join']['y']); reg(('p', y), nodes[y]['t'], [nodes[y][c] for c in 'zyx'])
            last = ('d', int(steps[-1]['x'])); K = len(steps)
            back = ([wpos[K - 1 - i] for i in range(K)] + list(tr))[:4]
            yc = chain(y, out, pos, 4) if fwd else chain(y, prev, pos, 4)
            if fwd: items.append(('%d_1_j' % n, last, ('p', y), back, list(yc), ('p', n), last))
            else: items.append(('%d_0_j' % n, ('p', y), last, list(yc), back, ('p', n), last))
        if w['near']:
            s = w['near']; nk = ('d', int(s['x'])); reg(nk, s['t'], [s['z'], s['y'], s['xx']])
            xp = [np.array([s['z'], s['y'], s['xx']], np.float32) * S]
            if fwd: items.append(('%d_1_n' % n, ('p', n), nk, list(tr), xp, ('p', n), nk))
            else: items.append(('%d_0_n' % n, nk, ('p', n), xp, list(tr), ('p', n), nk))
        # reference edge of the terminus track (its last real link)
        if len(tr) >= 2:
            m2 = prev.get(n) if fwd else (out.get(n, [None])[0] if len(out.get(n, [])) == 1 else None)
            if m2 is not None:
                reg(('p', m2), nodes[m2]['t'], [nodes[m2][c] for c in 'zyx'])
                h2 = chain(m2, prev, pos, 4) if fwd else [pos[n]]
                if fwd: items.append(('%d_1_ref' % n, ('p', m2), ('p', n), list(chain(m2, prev, pos, 4)), [pos[n]], ('p', n), ('p', n)))
                else: items.append(('%d_0_ref' % n, ('p', n), ('p', m2), [pos[n]], list(chain(m2, out, pos, 4)), ('p', n), ('p', n)))
    keys = sorted(need, key=lambda k: need[k][0])
    lookup = {k: i for i, k in enumerate(keys)}
    movie = Movie('/workspace/data/train/%s.zarr' % name, context=5)
    embs = []; inten = []
    with torch.inference_mode():
        for s0 in range(0, len(keys), 256):
            ch = keys[s0:s0 + 256]
            x = movie.patches([need[k][0] for k in ch], [need[k][1] for k in ch])
            xt = torch.from_numpy(x.astype(np.float32))
            embs.append(model.encode(xt).numpy())
            c = x[:, 2, 5:8, 9:15, 9:15].astype(np.float32)
            inten.append(np.stack([c.mean((1, 2, 3)), c.std((1, 2, 3))], 1))
    E = np.concatenate(embs) if embs else np.zeros((0, 64), np.float32); I = np.concatenate(inten) if inten else np.zeros((0, 2))
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-6)
    res = {}
    if items:
        geo = np.stack([edge_geometry(np.asarray(h, np.float32), np.asarray(fu, np.float32)) for _, _, _, h, fu, _, _ in items])
        a = torch.from_numpy(E[[lookup[i[1]] for i in items]]); b = torch.from_numpy(E[[lookup[i[2]] for i in items]])
        with torch.inference_mode():
            pr = torch.sigmoid(model.edge_logits(a, b, torch.from_numpy(geo))).numpy()
        ref = {}
        for (key, sk, dk, h, fu, tk, nk), p in zip(items, pr):
            if key.endswith('_ref'): ref[key[:-4]] = float(p)
        for (key, sk, dk, h, fu, tk, nk), p in zip(items, pr):
            if key.endswith('_ref'): continue
            si, di, ti, ni = lookup[sk], lookup[dk], lookup[tk], lookup[nk]
            pre = '_'.join(key.split('_')[:2])
            res[key] = [round(float(p), 5), round(float(En[si] @ En[di]), 4), round(float(En[ti] @ En[ni]), 4),
                        round(float((I[ni, 0] + 1e-3) / (I[ti, 0] + 1e-3)), 4), round(float((I[ni, 1] + 1e-3) / (I[ti, 1] + 1e-3)), 4),
                        ref.get(pre, float('nan'))]
    return name, res, len(keys), round(time.time() - t0, 1)


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/nm/ver_cands/*/*.json'))
    if len(sys.argv) > 1: files = files[:int(sys.argv[1])]
    out = {}
    with Pool(int(os.environ.get('RULE_POOL', '24')), maxtasksperchild=2) as p:
        for name, res, nk, dt in p.imap_unordered(job, files):
            out[name] = res; print(name, nk, dt, flush=True)
    json.dump(out, open('/workspace/cl/nm/ver_b1feat%s.json' % ('_test' if len(sys.argv) > 1 else ''), 'w'))
    print('done', len(out))
