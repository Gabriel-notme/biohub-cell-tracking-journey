"""Same nucleus or neighbouring nucleus? Intensity profile along the segment pred P -> GT G (3x5x5 box means, 13 samples) for
loc_an records; dip = min(profile) / min(I(P), I(G)). Also montage PNGs of random U examples (xy at z_P, xy at z_G, xz)."""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
R = json.load(open('/workspace/cl/nm/loc_an_rows.json'))
by = defaultdict(list)
for i, r in enumerate(R): by[r['m']].append(i)


def job(m):
    import numcodecs.blosc; numcodecs.blosc.use_threads = False
    import zarr
    a = zarr.open_group('/workspace/data/train/%s.zarr' % m, mode='r')['0']
    out = {}; cache = {}
    for i in sorted(by[m], key=lambda i: R[i]['t']):
        r = R[i]; t = r['t']
        if t not in cache:
            cache.clear(); cache[t] = np.asarray(a[t]).astype(np.float32)
        f = cache[t]
        P = np.array(r['P']) / S; G = np.array(r['G']) / S
        prof = []
        for s in np.linspace(0, 1, 13):
            q = np.round(P + s * (G - P)).astype(int); z, y, x = [int(np.clip(v, 0, n - 1)) for v, n in zip(q, f.shape)]
            prof.append(float(f[max(0, z - 1):z + 2, max(0, y - 2):y + 3, max(0, x - 2):x + 3].mean()))
        prof = np.array(prof)
        out[i] = dict(dip=float(prof.min() / max(1e-6, min(prof[0], prof[-1]))), ip=float(prof[0]), ig=float(prof[-1]), prof=prof.round(1).tolist())
    return out


if __name__ == '__main__':
    with Pool(24, maxtasksperchild=4) as p: outs = p.map(job, sorted(by), chunksize=1)
    D = {}
    for o in outs: D.update(o)
    rows = [dict(R[i], **D[i]) for i in sorted(D)]
    json.dump(rows, open('/workspace/cl/nm/loc_dip_rows.json', 'w'))
    for emb in ['44b6', '6bba']:
        for typ in ['U', 'M57', 'M']:
            X = [r for r in rows if r['typ'] == typ and r['m'].startswith(emb)]
            dip = np.array([r['dip'] for r in X])
            dd = np.array([r['drop_g'] if r['drop_g'] is not None else 99 for r in X])
            print('%s %-4s n=%4d dip med %.2f q25 %.2f | frac dip<0.8 %.3f  <0.6 %.3f | among dip<0.8: dropped det<=3um of GT %.3f | dip>=0.8: %.3f' % (
                emb, typ, len(X), np.median(dip), np.quantile(dip, .25), (dip < .8).mean(), (dip < .6).mean(),
                (dd[dip < .8] <= 3).mean() if (dip < .8).any() else 0, (dd[dip >= .8] <= 3).mean()))
