"""Probe: D4 test-time augmentation of the b1 fork head on the forks of P13 outputs. Would a removal-only rule
("drop a division-completion fork whose TTA-averaged probability falls below the acceptance threshold") remove FPs without TPs?
usage: tta_forks.py <gpu> <sets>  -> /workspace/cl/ttaf/<set>__<movie>.json"""
import os, sys, json, glob
gpu = sys.argv[1]; os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ.setdefault('POLARS_MAX_THREADS', '2')
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
OUT = Path('/workspace/cl/ttaf'); OUT.mkdir(exist_ok=True)


def tf(x, i):
    if i == 0: return x
    if i <= 3: return x.flip([(-1,), (-2,), (-2, -1)][i - 1])
    if i <= 5: return torch.rot90(x, [1, 3][i - 4], (-2, -1))
    if i == 6: return x.transpose(-1, -2)
    return torch.rot90(x, 1, (-2, -1)).transpose(-1, -2)


def main():
    from cell_event import SCALE, Movie, chain, fork_geometry, load_event_model
    import evalx
    from tracking_cellmot.division_metrics import score_divisions
    model, _ = load_event_model(Path(ART) / 'b1_best.pt', 'cuda')
    for s in sys.argv[2].split(','):
        for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s)):
            name = Path(f).stem; of = OUT / ('%s__%s.json' % (s, name))
            if of.exists(): continue
            nodes, edges = evalx.load_graph_json(f)
            out, prev = defaultdict(list), {}
            eat = {}
            for e in edges:
                a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a; eat[(a, b)] = e
            forks = [n for n in out if len(out[n]) == 2]
            rows = []
            if forks:
                pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
                trip = [(p, out[p][0], out[p][1]) for p in forks]
                need = sorted({x for t in trip for x in t}, key=lambda n: (nodes[n]['t'], n)); lk = {n: i for i, n in enumerate(need)}
                mv = Movie(Path('/workspace/data/train') / (name + '.zarr'), context=5)
                X = torch.from_numpy(mv.patches([nodes[n]['t'] for n in need], [[nodes[n][k] for k in 'zyx'] for n in need])).to('cuda', dtype=torch.float32)
                geo = torch.from_numpy(np.stack([fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in trip]).astype(np.float32)).to('cuda')
                ix = np.array([[lk[p], lk[a], lk[b]] for p, a, b in trip])
                L = []
                with torch.inference_mode():
                    for i in range(8):
                        E = torch.cat([model.encode(tf(X[j:j + 256], i)).float() for j in range(0, len(X), 256)])
                        L.append(model.fork_logits(E[ix[:, 0]], E[ix[:, 1]], E[ix[:, 2]], geo).float().cpu().numpy())
                L = np.stack(L)
                gt, _ = evalx.load_gt(name)
                pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
                res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
                tp = {inv[int(x)] for x in res.tp_forks}; fpk = {inv[int(x)] for x in res.fp_forks}
                for j, (p, a, b) in enumerate(trip):
                    dc = bool(eat[(p, a)].get('div_complete') or eat[(p, b)].get('div_complete'))
                    rows.append(dict(p=p, dc=dc, base=float(1 / (1 + np.exp(-L[0, j]))), tta=float(1 / (1 + np.exp(-L[:, j].mean()))),
                                     tmin=float(1 / (1 + np.exp(-L[:, j].min()))), lab='TP' if p in tp else ('FP' if p in fpk else 'U')))
            of.write_text(json.dumps(rows))
    print('DONE', gpu, flush=True)


if __name__ == '__main__':
    main()
