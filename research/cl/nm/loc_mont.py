"""Montage of random U / M57 examples: per row: xy slice at z_P (red=P, green=G), xy slice at z_G, xz slice at y_P, zoomed 32 um box.
Also other pred nodes (blue) and all GT nodes (yellow) at that frame within the box."""
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import numcodecs.blosc; numcodecs.blosc.use_threads = False
import zarr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
S = np.array([1.625, .40625, .40625])
R = json.load(open('/workspace/cl/nm/loc_dip_rows.json'))
typ = sys.argv[1]; seed = int(sys.argv[2]); n = 10
rng = np.random.default_rng(seed)
X = [r for r in R if r['typ'] == typ]
X = [X[i] for i in rng.choice(len(X), n, replace=False)]
fig, ax = plt.subplots(n, 4, figsize=(12, 3 * n))
H = 40  # half box in xy px (16 um); z half 10 slices
for k, r in enumerate(X):
    a = zarr.open_group('/workspace/data/train/%s.zarr' % r['m'], mode='r')['0']
    f = np.asarray(a[r['t']]).astype(np.float32)
    P = np.round(np.array(r['P']) / S).astype(int); G = np.round(np.array(r['G']) / S).astype(int)
    c = ((P + G) / 2).astype(int)
    y0, x0 = c[1] - H, c[2] - H
    lo, hi = np.percentile(f, [1, 99.7])
    for j, zz in enumerate([P[0], G[0]]):
        zz = int(np.clip(zz, 0, f.shape[0] - 1))
        im = f[zz, max(0, y0):y0 + 2 * H, max(0, x0):x0 + 2 * H]
        ax[k, j].imshow(im, cmap='gray', vmin=lo, vmax=hi)
        ax[k, j].plot(P[2] - max(0, x0), P[1] - max(0, y0), 'r+', ms=12, mew=2); ax[k, j].plot(G[2] - max(0, x0), G[1] - max(0, y0), 'gx', ms=12, mew=2)
        ax[k, j].set_title('%s t%d z=%d (%s)' % (r['m'][:9], r['t'], zz, 'zP' if j == 0 else 'zG'), fontsize=8); ax[k, j].axis('off')
    yy = int(np.clip(c[1], 0, f.shape[1] - 1))
    im = f[:, yy, max(0, x0):x0 + 2 * H]
    ax[k, 2].imshow(im, cmap='gray', vmin=lo, vmax=hi, aspect=1.625 / .40625)
    ax[k, 2].plot(P[2] - max(0, x0), P[0], 'r+', ms=12, mew=2); ax[k, 2].plot(G[2] - max(0, x0), G[0], 'gx', ms=12, mew=2)
    ax[k, 2].set_title('xz at y=%d  d=%.1f dz=%.1f' % (yy, r['d'], r['dz']), fontsize=8); ax[k, 2].axis('off')
    ax[k, 3].plot(r['prof']); ax[k, 3].set_title('profile P->G dip %.2f drop_g %.1f other %.1f' % (r['dip'], r['drop_g'] or 99, r['other_d']), fontsize=8)
plt.tight_layout(); plt.savefig('/workspace/cl/nm/loc_mont_%s_%d.png' % (typ, seed), dpi=60)
print('ok')
