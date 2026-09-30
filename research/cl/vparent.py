"""Virtual-parent re-scoring of 'stolen' division candidates. b1's fork head was trained with the parent patch centred on the GT parent
(between the daughters). When the detector split a dividing nucleus into p and q, p sits on one side and b1 sees an off-centre parent.
Re-score with a virtual parent at the p/q midpoint (patch + history geometry). Labelled candidates only (P/D vs subsampled N).
usage: vparent.py <gpu> <sets>  -> /workspace/cl/vp/<set>__<movie>.json"""
import os, sys, json, glob
gpu = sys.argv[1]; os.environ['CUDA_VISIBLE_DEVICES'] = gpu
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, ART); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
OUT = Path('/workspace/cl/vp'); OUT.mkdir(exist_ok=True)
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def main():
    from cell_event import SCALE, Movie, chain, fork_geometry, load_event_model
    import evalx
    model, _ = load_event_model(Path(ART) / 'b1_best.pt', 'cuda')
    rng = np.random.default_rng(0)
    for s in sys.argv[2].split(','):
        for f in sorted(glob.glob('/workspace/cl/cl2/%s/*.json' % s)):
            name = Path(f).stem; of = OUT / ('%s__%s.json' % (s, name))
            if of.exists(): continue
            labs = [r for r in json.load(open(f)) if r['typ'] == 'stolen' and r['lab'] in ('P', 'D', 'N')]
            dump = Path('/workspace/cl/cands_p8/%s/%s.json' % (s, name))
            base = {(c['p'], c['a'], c['b']): c['fork'] for c in json.load(open(dump))} if dump.exists() else {}
            pos_ = [r for r in labs if r['lab'] != 'N']; neg = [r for r in labs if r['lab'] == 'N']
            keep = pos_ + ([neg[i] for i in rng.choice(len(neg), min(len(neg), 150), replace=False)] if neg else [])
            if not keep: of.write_text('[]'); continue
            nodes, edges = evalx.load_graph_json(Path(B5[s]) / (name + '.json'))
            out, prev = defaultdict(list), {}
            for e in edges:
                a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
            pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
            raw = {n: [v['z'], v['y'], v['x']] for n, v in nodes.items()}
            mv = Movie(Path('/workspace/data/train') / (name + '.zarr'), context=5)
            # patches: a, b at t+1 (node coords); virtual parent at t = midpoint(p, q)
            T, C, G = [], [], []
            for r in keep:
                p, a, b, q = r['p'], r['a'], r['b'], r['q']
                vm = [(x + y) / 2 for x, y in zip(raw[p], raw[q])]
                t = int(nodes[p]['t'])
                T += [t, t + 1, t + 1]; C += [vm, raw[a], raw[b]]
                hp = chain(p, prev, pos); hq = chain(q, prev, pos); k = min(len(hp), len(hq))
                hist = (hp[:k] + hq[:k]) / 2 if k > 0 else hp
                G.append(fork_geometry(hist, chain(a, out, pos), chain(b, out, pos)))
            X = torch.from_numpy(mv.patches(T, C)).to('cuda', dtype=torch.float32)
            with torch.inference_mode():
                E = torch.cat([model.encode(X[i:i + 384]).float() for i in range(0, len(X), 384)])
                E = E.view(len(keep), 3, -1)
                lg = model.fork_logits(E[:, 0], E[:, 1], E[:, 2], torch.from_numpy(np.stack(G).astype(np.float32)).cuda()).float().cpu().numpy()
            res = []
            for r, l in zip(keep, lg):
                res.append(dict(lab=r['lab'], base=base.get((r['p'], r['a'], r['b'])), vp=float(1 / (1 + np.exp(-l))), q_hist=r.get('q_hist'), d_qp=r.get('d_qp'),
                                n_neg_total=len(neg)))
            of.write_text(json.dumps(res))
    print('DONE', gpu, flush=True)


if __name__ == '__main__':
    main()
