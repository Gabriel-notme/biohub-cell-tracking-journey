"""Per-track image features for junk pruning (same track order as jprune.features): node intensity (3x7x7 box, movie-quantile
normalised), local contrast (box minus surrounding shell), aggregated per track (mean/min/std).
usage: jp_img.py <tag> <graph_dir> -> /workspace/cl/jp/<tag>_img.pkl  {movie: array (n_tracks, 6)}"""
import os, sys, glob, pickle, json
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
IMGF = ['I_mean', 'I_min', 'I_std', 'C_mean', 'C_min', 'C_std']


def node_feats(arr, lo, hi, nodes, ids):
    by_t = defaultdict(list)
    for n in ids: by_t[int(nodes[n]['t'])].append(n)
    I = {}; Cc = {}
    for t, ns in by_t.items():
        F = (np.asarray(arr[t], np.float32) - lo) / (hi - lo + 1e-6)
        Z, Y, X = F.shape
        # integral-free box means via cumulative sums
        pad = np.pad(F, ((1, 1), (6, 6), (6, 6)), mode='edge')
        cs = pad.cumsum(0).cumsum(1).cumsum(2)
        cs = np.pad(cs, ((1, 0), (1, 0), (1, 0)))

        def boxmean(c, rz, ry):
            z0 = c[:, 0] - rz + 1; z1 = c[:, 0] + rz + 2; y0 = c[:, 1] - ry + 6; y1 = c[:, 1] + ry + 7; x0 = c[:, 2] - ry + 6; x1 = c[:, 2] + ry + 7
            z0 = np.clip(z0, 0, Z + 2); z1 = np.clip(z1, 0, Z + 2); y0 = np.clip(y0, 0, Y + 12); y1 = np.clip(y1, 0, Y + 12); x0 = np.clip(x0, 0, X + 12); x1 = np.clip(x1, 0, X + 12)
            s = (cs[z1, y1, x1] - cs[z0, y1, x1] - cs[z1, y0, x1] - cs[z1, y1, x0] + cs[z0, y0, x1] + cs[z0, y1, x0] + cs[z1, y0, x0] - cs[z0, y0, x0])
            v = (z1 - z0) * (y1 - y0) * (x1 - x0)
            return s / np.maximum(v, 1), s, v
        c = np.array([[int(round(nodes[n][k])) for k in 'zyx'] for n in ns])
        c = np.clip(c, 0, [Z - 1, Y - 1, X - 1])
        m_in, s_in, v_in = boxmean(c, 1, 3)
        m_out, s_out, v_out = boxmean(c, 1, 6)
        shell = (s_out - s_in) / np.maximum(v_out - v_in, 1)
        for n, a, b in zip(ns, m_in, m_in - shell): I[n] = float(a); Cc[n] = float(b)
    return I, Cc


def job(a):
    tag, f = a
    name = Path(f).stem
    import evalx, jprune, zarr
    nodes, edges = evalx.load_graph_json(f)
    trk, _ = jprune.tracks(nodes, edges)
    g = zarr.open_group('/workspace/data/train/%s.zarr' % name, mode='r'); arr = g['0']
    qs = g.attrs.get('image_statistics', {}).get('quantiles', {})
    lo = float(qs.get('0.001', 0)); hi = float(qs.get('0.999', 1))
    if hi <= lo:
        v = np.asarray(arr[0])[::2, ::4, ::4]; lo, hi = np.percentile(v, [.1, 99.9])
    ids = [n for c in trk for n in c]
    I, Cc = node_feats(arr, lo, hi, nodes, ids)
    out = np.array([[np.mean([I[n] for n in c]), np.min([I[n] for n in c]), np.std([I[n] for n in c]),
                     np.mean([Cc[n] for n in c]), np.min([Cc[n] for n in c]), np.std([Cc[n] for n in c])] for c in trk], float) if trk else np.zeros((0, 6))
    return name, out


if __name__ == '__main__':
    tag, gdir = sys.argv[1], sys.argv[2]
    with Pool(48) as p: R = dict(p.map(job, [(tag, f) for f in sorted(glob.glob(gdir + '/*.json'))]))
    pickle.dump(R, open('/workspace/cl/jp/%s_img.pkl' % tag, 'wb'))
    print(tag, 'movies', len(R), 'tracks', sum(len(v) for v in R.values()))
