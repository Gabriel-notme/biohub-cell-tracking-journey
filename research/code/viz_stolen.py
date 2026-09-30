import json, sys, os
import numpy as np, zarr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/code')
rows = json.load(open('/workspace/runs/divtest_hold36/labeled.json'))
pos = [r for r in rows if r['lab'] == 'pos']
gdir = sys.argv[1]
os.makedirs('/workspace/viz', exist_ok=True)
H = 40
fig, axes = plt.subplots(len(pos), 4, figsize=(12, 3 * len(pos)))
for i, r in enumerate(pos):
    g = json.load(open('%s/%s.json' % (gdir, r['movie'])))
    N = {int(k): v for k, v in g['nodes'].items()}
    p, a, b = N[r['p']], N[r['a']], N[r['b']]; q = N.get(r['q']) if r.get('q') is not None else None
    arr = zarr.open('/workspace/data/train/%s.zarr/0' % r['movie'], mode='r')
    t = int(p['t']); cy, cx, cz = int(round(p['y'])), int(round(p['x'])), int(round(p['z']))
    for j, tt in enumerate([t - 1, t, t + 1, t + 2]):
        ax = axes[i, j]; ax.axis('off')
        if tt < 0 or tt >= arr.shape[0]: continue
        img = np.asarray(arr[tt, max(0, cz - 4):cz + 5, max(0, cy - H):cy + H, max(0, cx - H):cx + H]).max(0).astype(np.float32)
        ax.imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
        y0, x0 = max(0, cy - H), max(0, cx - H)
        for n, v in N.items():
            if int(v['t']) == tt and abs(v['y'] - cy) < H and abs(v['x'] - cx) < H and abs(v['z'] - cz) < 6:
                col = 'yellow'; ms = 3
                if n == r['p']: col, ms = 'red', 7
                elif r.get('q') is not None and n == r['q']: col, ms = 'lime', 7
                elif n == r['a']: col, ms = 'cyan', 7
                elif n == r['b']: col, ms = 'magenta', 7
                ax.plot(v['x'] - x0, v['y'] - y0, 'o', mfc='none', mec=col, ms=ms)
        ax.set_title('%s t=%d fork=%.3f' % (r['movie'][-8:], tt, r['fork']) if j == 0 else 't=%d' % tt, fontsize=8)
plt.tight_layout(); plt.savefig('/workspace/viz/stolen_hold36.png', dpi=60)
print('ok', len(pos))
