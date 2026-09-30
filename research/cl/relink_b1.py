"""Relink candidates on G0 (pre-link graphs) + b1 edge-head probabilities for the proposed edge (s->d) and the edges it would replace
(s->cur_d, cur_s->d). usage: relink_b1.py <gpu> <sets>  -> /workspace/cl/lo/rb1/<set>__<movie>.pkl (ids, X_base, b1 feats, label)"""
import os, sys, json, glob, pickle
gpu = sys.argv[1]; os.environ['CUDA_VISIBLE_DEVICES'] = gpu
os.environ.setdefault('POLARS_MAX_THREADS', '2')
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
import numpy as np
OUT = Path('/workspace/cl/lo/rb1'); OUT.mkdir(exist_ok=True, parents=True)
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def main():
    from refine_events import EventRefiner
    from cell_event import SCALE, chain, edge_geometry
    import evalx, relink, edge_link
    from loeo import gt_maps
    er = EventRefiner([Path(ART) / 'b1_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, '/workspace/cl/rb1_cache_%s' % gpu)
    for s in sys.argv[2].split(','):
        for f in sorted(glob.glob('/workspace/cl/ps_pre_%s/graphs/*.json' % s)):
            name = Path(f).stem; of = OUT / ('%s__%s.pkl' % (s, name))
            if of.exists(): continue
            nodes, edges = evalx.load_graph_json(f)
            rows = relink.features(nodes, edges, edge_link.load_full(Path(FULL[s]) / (name + '.geff')))
            if not rows:
                pickle.dump(dict(rows=[], X=np.zeros((0, len(relink.FEATS))), B=np.zeros((0, 5)), y=np.zeros(0)), open(of, 'wb')); continue
            out, prev = defaultdict(list), {}
            for e in edges:
                a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
            pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
            pairs = set()
            for r in rows:
                pairs.add((r['s'], r['d']))
                if r.get('cur_d') is not None: pairs.add((r['s'], r['cur_d']))
                if r.get('cur_s') is not None: pairs.add((r['cur_s'], r['d']))
            pairs = sorted(pairs)
            need = {x for pr in pairs for x in pr}
            ids, emb = er.embeddings(name, {n: nodes[n] for n in need}); lk = {n: i for i, n in enumerate(ids)}
            geo = [edge_geometry(chain(a, prev, pos), chain(b, out, pos)) for a, b in pairs]
            pr = er.score('edge', pairs, geo, emb, lk); P = dict(zip(pairs, map(float, pr)))
            B = []
            for r in rows:
                sd = P[(r['s'], r['d'])]; sc = P.get((r['s'], r['cur_d']), -1.) if r.get('cur_d') is not None else -1.
                cd = P.get((r['cur_s'], r['d']), -1.) if r.get('cur_s') is not None else -1.
                B.append([sd, sc, cd, sd - sc if sc >= 0 else 1., sd - cd if cd >= 0 else 1.])
            p2g, gs, gp = gt_maps(name, nodes, edges)
            y = []
            for r in rows:
                a_, d_ = p2g.get(r['s']), p2g.get(r['d'])
                valid = (a_ is not None and len(gs.get(a_, [])) > 0) or (d_ is not None and d_ in gp)
                ok = a_ is not None and d_ is not None and d_ in gs.get(a_, [])
                y.append(1 if ok else (0 if valid else -1))
            X = np.array([[r.get(k, -1) if r.get(k) is not None else -1 for k in relink.FEATS] for r in rows], np.float32)
            pickle.dump(dict(X=X, B=np.array(B, np.float32), y=np.array(y)), open(of, 'wb'))
            for q in Path('/workspace/cl/rb1_cache_%s' % gpu).glob('*%s*' % name): q.unlink()
    print('DONE', gpu, flush=True)


if __name__ == '__main__':
    main()
