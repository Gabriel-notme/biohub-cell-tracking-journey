"""Does B5's centroid CNN (in-sample, trained on offsets <= 6.5 um) point U nodes toward their GT node?
phase 'extract' (CPU pool): 7x12x24x24 patches at the current pred position of every loc_an record -> /workspace/cl/nm/loc_patches/<movie>.npy
phase 'infer' (one GPU, short): delta/error per record -> loc_cen_rows.json; summary per type/embryo."""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/art_b56/artifact_bundle')
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
OUT = '/workspace/cl/nm/loc_patches'
R = json.load(open('/workspace/cl/nm/loc_an_rows.json'))
by = defaultdict(list)
for i, r in enumerate(R): by[r['m']].append(i)


def ext(m):
    import numcodecs.blosc; numcodecs.blosc.use_threads = False
    from cell_event import Movie
    ix = by[m]; mv = Movie('/workspace/data/train/%s.zarr' % m, 7)
    coords = np.array([np.array(R[i]['P']) / S for i in ix]); times = [R[i]['t'] for i in ix]
    x = mv.patches(times, coords); np.save('%s/%s.npy' % (OUT, m), x); return m


if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        os.makedirs(OUT, exist_ok=True)
        with Pool(24, maxtasksperchild=4) as p: p.map(ext, sorted(by), chunksize=1)
        print('extracted', len(by))
    else:
        import torch
        from centroid_model import load_centroid
        from cell_event import STRIDE, SCALE
        model, ck = load_centroid('/workspace/art_b56/artifact_bundle/centroid_v1_frozen.pt')
        delta = np.zeros((len(R), 3), np.float32); err = np.zeros(len(R), np.float32)
        with torch.inference_mode():
            for m, ix in sorted(by.items()):
                x = np.load('%s/%s.npy' % (OUT, m))
                coords = np.array([np.array(R[i]['P']) / S for i in ix])
                frac = (coords - np.rint(coords / STRIDE) * STRIDE) * SCALE
                for a in range(0, len(ix), 256):
                    xx = torch.from_numpy(x[a:a + 256]).cuda().float(); ff = torch.from_numpy(frac[a:a + 256].astype(np.float32)).cuda()
                    with torch.autocast('cuda', dtype=torch.float16): d, e = model(xx, ff)
                    delta[ix[a:a + 256]] = d.float().cpu().numpy(); err[ix[a:a + 256]] = e.float().cpu().numpy()
        out = []
        for r, d, e in zip(R, delta, err):
            P = np.array(r['P']); G = np.array(r['G'])
            out.append(dict(typ=r['typ'], m=r['m'], d0=r['d'], d_full=float(np.linalg.norm(P + d - G)), d_half=float(np.linalg.norm(P + 0.5 * d - G)),
                            err=float(e), dlen=float(np.linalg.norm(d)), cos=float(np.dot(d, G - P) / (np.linalg.norm(d) * np.linalg.norm(G - P) + 1e-9)), delta=d.tolist()))
        json.dump(out, open('/workspace/cl/nm/loc_cen_rows.json', 'w'))
        for emb in ['44b6', '6bba']:
            for typ in ['U', 'M57', 'M']:
                X = [o for o in out if o['typ'] == typ and o['m'].startswith(emb)]
                a = lambda k: np.array([o[k] for o in X])
                print('%s %-4s n=%4d d0 med %.2f | full: d med %.2f, <=7 %.3f, >7 %.3f | half: <=7 %.3f | err med %.2f | |delta| med %.2f | cos med %.2f | frac err<=2.5 %.3f' % (
                    emb, typ, len(X), np.median(a('d0')), np.median(a('d_full')), (a('d_full') <= 7).mean(), (a('d_full') > 7).mean(), (a('d_half') <= 7).mean(),
                    np.median(a('err')), np.median(a('dlen')), np.median(a('cos')), (a('err') <= 2.5).mean()))
