"""Render missed 'stolen'-type divisions (P13 outputs, b1-unseen hold36) as local z-max projections over frames t-3..t+3.
GT parent/children in red/yellow, predicted nodes in cyan (p's track), magenta (the stealing track q -> b). One PNG per case."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
import numpy as np
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S = np.array([1.625, 0.40625, 0.40625])
OUT = Path('/workspace/cl/viz'); OUT.mkdir(exist_ok=True)
R = 40  # half crop in pixels (~16 um)


def cases(name):
    import evalx
    from tracking_cellmot.division_metrics import score_divisions, _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json('/workspace/cl/ps_p13_hold36/graphs/%s.json' % name)
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    g2p = {b: a for a, b in p2g.items()}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): (int(t), np.array([z, y, x], float)) for i, t, z, y, x in zip(*[ga[k].to_list() for k in [K.NODE_ID, 't', 'z', 'y', 'x']])}
    ea = gt.edge_attrs(); gch = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gch[int(a)].append(int(b)); gpar[int(b)] = int(a)
    ch = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); ch[a].append(b); par[b] = a
    out = []
    for d, v in res.scores.items():
        d = int(d)
        if v or len(gch[d]) < 2: continue
        p = g2p.get(d); km = [g2p.get(c) for c in gch[d]]
        if p is None or any(x is None for x in km): continue
        cont = [x for x in km if par.get(x) == p]; other = [x for x in km if par.get(x) != p]
        if len(cont) != 1 or len(other) != 1 or par.get(other[0]) is None: continue
        out.append(dict(d=d, p=p, a=cont[0], b=other[0], q=par[other[0]]))
    return nodes, par, ch, gpos, gch, gpar, out


def render(name, nodes, par, ch, gpos, gch, gpar, c, k):
    arr = zarr.open_group('/workspace/data/train/%s.zarr' % name, mode='r')['0']
    td, pd = gpos[c['d']]
    cz, cy, cx = [int(round(x)) for x in pd]
    frames = list(range(td - 3, td + 4))
    fig, axs = plt.subplots(1, len(frames), figsize=(2.2 * len(frames), 2.6))
    # predicted track chains through p (backwards/forwards) and through q->b
    def chain_nodes(start, back=True):
        out = []; x = start
        for _ in range(6):
            out.append(x)
            if back: x = par.get(x)
            else: x = ch[x][0] if len(ch.get(x, [])) >= 1 else None
            if x is None: break
        return out
    ptrack = set(chain_nodes(c['p']) + chain_nodes(c['a'], back=False))
    qtrack = set(chain_nodes(c['q']) + chain_nodes(c['b'], back=False))
    gnodes = defaultdict(list)
    x = c['d']; lin = [x]
    for _ in range(4):
        if x in gpar: x = gpar[x]; lin.append(x)
    fr = list(gch[c['d']])
    for _ in range(3):
        nxt = [y for z in fr for y in gch[z]]; lin += fr; fr = nxt
    for g in lin: gnodes[gpos[g][0]].append((g, gpos[g][1]))
    for ax, t in zip(axs, frames):
        if t < 0 or t >= arr.shape[0]: ax.axis('off'); continue
        vol = np.asarray(arr[t, max(0, cz - 3):cz + 4, max(0, cy - R):cy + R, max(0, cx - R):cx + R], np.float32)
        img = vol.max(0); lo, hi = np.percentile(img, [1, 99.7]); ax.imshow(np.clip((img - lo) / (hi - lo + 1e-6), 0, 1), cmap='gray')
        oy, ox = max(0, cy - R), max(0, cx - R)
        for g, gp in gnodes.get(t, []):
            col = 'red' if g == c['d'] or t < td else 'yellow'
            ax.scatter([gp[2] - ox], [gp[1] - oy], s=60, facecolors='none', edgecolors=col, linewidths=1.5)
        for n, v in nodes.items():
            if int(v['t']) != t: continue
            if abs(v['y'] - cy) > R or abs(v['x'] - cx) > R: continue
            col = 'cyan' if n in ptrack else ('magenta' if n in qtrack else 'lime')
            ax.scatter([v['x'] - ox], [v['y'] - oy], s=12, c=col, marker='x')
        ax.set_title('t=%d%s' % (t, ' (GT div)' if t == td else ''), fontsize=8); ax.axis('off')
    fig.suptitle('%s GT div %d at t=%d  | red=GT parent, yellow=GT daughters, cyan=pred p-track, magenta=pred q-track (stole daughter), lime=other' % (name, c['d'], td), fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / ('case%02d_%s.png' % (k, name)), dpi=90); plt.close(fig)


if __name__ == '__main__':
    k = 0
    for f in sorted(glob.glob('/workspace/cl/ps_p13_hold36/graphs/*.json')):
        name = Path(f).stem
        nodes, par, ch, gpos, gch, gpar, cs = cases(name)
        for c in cs:
            render(name, nodes, par, ch, gpos, gch, gpar, c, k); k += 1
            if k >= 8: break
        if k >= 8: break
    print('rendered', k)
