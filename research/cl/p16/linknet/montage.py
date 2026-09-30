"""Montage of top-scoring held-out rows (by ln_A1): frames t-1..t+2, z-max-projection (+-2 slices) around the proposed link.
Markers: P15 nodes (gray), s + its history (cyan), proposed d (green), current child c (red), q = current parent of d (orange),
GT nodes (white x; GT edges white lines).  usage: montage.py <eval pickle> <emb> <n> <out.png> [fp|tp]"""
import os, sys, pickle, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
EV, EMB, N, OUTP = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
MODE = sys.argv[5] if len(sys.argv) > 5 else 'fp'
R = pickle.load(open(EV, 'rb'))[EMB]
cand = []
for d in R:
    v = d['scv']['ln_A1']
    for j in np.argsort(-v)[:30]:
        if (MODE == 'fp' and d['y'][j] == 0) or (MODE == 'tp' and d['y'][j] == 1): cand.append((v[j], d['set'], d['movie'], j))
cand.sort(reverse=True); cand = cand[:N]
import evalx
fig, axs = plt.subplots(len(cand), 4, figsize=(12, 3 * len(cand)))
W = 32  # half window in pooled px (26 um)
for r, (v, s, m, j) in enumerate(cand):
    d = [x for x in R if x['movie'] == m][0]
    tb = pickle.load(open('/workspace/cl/p16/linknet/tab/%s__%s.pkl' % (s, m), 'rb'))
    vol = np.load('/dev/shm/linknet/vol/%s.npy' % m, mmap_mode='r')
    zyx = tb['zyx'].copy(); zyx[:, 1:] = (zyx[:, 1:] - .5) / 2; tt = tb['t']
    S_, D_, C_, Q_ = d['S'][j], d['D'][j], d['C'][j], d['Q'][j]
    t = tt[S_]; cz, cy, cx = (zyx[S_] + zyx[D_]) / 2
    gt, _ = evalx.load_gt(m); K = evalx.K
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gid = np.array(ga[K.NODE_ID].to_list()); gT = np.array(ga['t'].to_list()); gZ = np.array(ga['z'].to_list()); gY = (np.array(ga['y'].to_list()) - .5) / 2; gX = (np.array(ga['x'].to_list()) - .5) / 2
    gi = {int(a): i for i, a in enumerate(gid)}
    ea = gt.edge_attrs(); ges = list(zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()))
    hist = [S_]; a = tb['par'][S_]
    while a >= 0 and len(hist) < 3: hist.append(a); a = tb['par'][a]
    for k, dt in enumerate((-1, 0, 1, 2)):
        ax = axs[r, k]; f = int(np.clip(t + dt, 0, vol.shape[0] - 1))
        z0, z1 = int(max(0, round(cz) - 2)), int(min(vol.shape[1], round(cz) + 3))
        y0, x0 = int(round(cy)) - W, int(round(cx)) - W
        img = np.zeros((2 * W, 2 * W), np.float32)
        ys, xs = slice(max(0, y0), min(vol.shape[2], y0 + 2 * W)), slice(max(0, x0), min(vol.shape[3], x0 + 2 * W))
        img[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = vol[f, z0:z1, ys, xs].max(0)
        ax.imshow(img, cmap='magma', vmin=0, vmax=np.percentile(img, 99.5) + 1)
        sel = (tt == f) & (np.abs(zyx[:, 0] - cz) < 4)
        ax.scatter(zyx[sel, 2] - x0, zyx[sel, 1] - y0, s=6, c='gray')
        for n_, col in [(h, 'cyan') for h in hist] + [(D_, 'lime'), (C_, 'red'), (Q_, 'orange')]:
            if n_ >= 0 and tt[n_] == f: ax.scatter(zyx[n_, 2] - x0, zyx[n_, 1] - y0, s=40, facecolors='none', edgecolors=col, linewidths=1.5)
        gs = (gT == f) & (np.abs(gZ - cz) < 4) & (np.abs(gY - cy) < W) & (np.abs(gX - cx) < W)
        ax.scatter(gX[gs] - x0, gY[gs] - y0, s=30, marker='x', c='white')
        for a_, b_ in ges:
            ia, ib = gi[int(a_)], gi[int(b_)]
            if gT[ia] == f - 1 and abs(gY[ia] - cy) < W and abs(gX[ia] - cx) < W and abs(gZ[ib] - cz) < 5:
                ax.plot([gX[ia] - x0, gX[ib] - x0], [gY[ia] - y0, gY[ib] - y0], c='white', lw=.8)
        ax.set_xlim(0, 2 * W); ax.set_ylim(2 * W, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('%s %s t%+d  pA %.2f y=%d dtp %+d dfp %+d' % (m[:9], s, dt, np.exp(v), d['y'][j], d['dtp'][j], d['dfp'][j]) if k == 0 else 't%+d (z %d-%d)' % (dt, z0, z1 - 1), fontsize=7)
plt.tight_layout(); plt.savefig(OUTP, dpi=80)
print('saved', OUTP, len(cand))
