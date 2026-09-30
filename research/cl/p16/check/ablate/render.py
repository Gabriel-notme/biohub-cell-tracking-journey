"""montage around a P15 node: columns t-1..t+3, rows: MIP raw | MIP + overlays | single z slice at 'drop' node + overlays.
usage: render.py <movie> <set> <center P15 node id> <json {label: [ids]}> <out.png> [title]"""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']: os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
import numpy as np, zarr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import evalx, tracksdata as td
K = td.DEFAULT_ATTR_KEYS
movie, sset, c0 = sys.argv[1], sys.argv[2], int(sys.argv[3]); H = json.loads(sys.argv[4]); out = sys.argv[5]; title = sys.argv[6] if len(sys.argv) > 6 else ''
n15, e15 = evalx.load_graph_json(os.environ.get('SRCDIR', '/workspace/cl/ps_p15_%s/graphs') % sset + '/%s.json' % movie)
n17, e17 = evalx.load_graph_json('/workspace/cl/p16/ps_p17_%s/graphs/%s.json' % (sset, movie))
gt, _ = evalx.load_gt(movie)
ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
G = {int(i): (int(t), float(z), float(y), float(x)) for i, t, z, y, x in zip(*[ga[c].to_list() for c in [K.NODE_ID, 't', 'z', 'y', 'x']])}
gea = gt.edge_attrs(); GE = list(zip(gea[K.EDGE_SOURCE].to_list(), gea[K.EDGE_TARGET].to_list()))
img = zarr.open_group('/workspace/data/train/%s.zarr' % movie, mode='r')['0']
T, Z, Y, X = img.shape
c = n15[c0]; t0 = int(c['t']); cy, cx = float(c['y']), float(c['x'])
allids = [c0] + [i for v in H.values() for i in v]
zs = [float(n15[i]['z']) for i in allids if i in n15]
z0, z1 = max(0, int(min(zs)) - 3), min(Z, int(max(zs)) + 4)
R = 55
y0, y1 = max(0, int(cy) - R), min(Y, int(cy) + R); x0, x1 = max(0, int(cx) - R), min(X, int(cx) + R)
frames = [t for t in range(t0 - 1, t0 + 4) if 0 <= t < T]
cols = {'parent': 'red', 'keep': 'lime', 'drop': 'magenta', 'lost': 'orange', 'other': 'yellow'}
dropz = int(round(np.mean([float(n15[i]['z']) for i in H.get('drop', H.get('lost', [c0])) if i in n15])))
fig, ax = plt.subplots(3, len(frames), figsize=(3.2 * len(frames), 9.6))
crops = {t: np.asarray(img[t, z0:z1, y0:y1, x0:x1]).astype(np.float32) for t in frames}
lo, hi = np.percentile(np.concatenate([v.max(0).ravel() for v in crops.values()]), [1, 99.7])
lab = {i: k for k, v in H.items() for i in v}; lab[c0] = lab.get(c0, 'parent')
for j, t in enumerate(frames):
    mip = crops[t].max(0); sl = crops[t][min(max(dropz - z0, 0), crops[t].shape[0] - 1)]
    for r, im in enumerate([mip, mip, sl]):
        a = ax[r, j]; a.imshow(np.clip((im - lo) / (hi - lo), 0, 1), cmap='gray', extent=(x0, x1, y1, y0)); a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title('t=%d  MIP z%d-%d' % (t, z0, z1 - 1), fontsize=9); continue
        if r == 2: a.set_title('z=%d slice' % dropz, fontsize=8)
        for n, v in n15.items():
            if int(v['t']) != t or not (y0 <= float(v['y']) < y1 and x0 <= float(v['x']) < x1) or not (z0 - 2 <= float(v['z']) < z1 + 2): continue
            k = lab.get(n); inz = abs(float(v['z']) - dropz) <= 2
            if r == 2 and not inz and k is None: continue
            gone = n not in n17
            a.plot(float(v['x']), float(v['y']), 'x' if gone else 'o', ms=7 if k else 4, mfc='none', mec=cols.get(k, 'deepskyblue' if not gone else 'gray'), mew=1.5)
            if k: a.text(float(v['x']) + 2, float(v['y']) - 2, '%s z%.0f' % (k[0], float(v['z'])), color=cols.get(k, 'w'), fontsize=6)
        for g, (gtt, gz, gy, gx) in G.items():
            if gtt == t and y0 <= gy < y1 and x0 <= gx < x1 and z0 - 3 <= gz < z1 + 3:
                a.plot(gx, gy, '+', color='cyan', ms=10, mew=1.2); a.text(gx - 8, gy + 7, 'g z%.0f' % gz, color='cyan', fontsize=6)
        for s_, d_ in GE:  # GT edge from t-1 to t drawn in frame t
            if G[d_][0] == t and y0 <= G[d_][2] < y1 and x0 <= G[d_][3] < x1:
                a.plot([G[s_][3], G[d_][3]], [G[s_][2], G[d_][2]], '-', color='cyan', lw=0.7)
        for e in e15:  # pred edges into frame t among labelled nodes
            s_, d_ = int(e['source_id']), int(e['target_id'])
            if int(n15[d_]['t']) == t and (s_ in lab or d_ in lab):
                a.plot([float(n15[s_]['x']), float(n15[d_]['x'])], [float(n15[s_]['y']), float(n15[d_]['y'])], '-', color=cols.get(lab.get(d_, lab.get(s_)), 'w'), lw=1)
fig.suptitle('%s %s  %s\nrow1 raw MIP | row2 MIP + P15 nodes (o kept in P17, x removed in P17; red parent, green kept daughter, magenta dropped branch, orange lost-TP nodes; cyan + = GT)' % (movie, sset, title), fontsize=9)
plt.tight_layout(); plt.savefig(out, dpi=90); print('saved', out)
