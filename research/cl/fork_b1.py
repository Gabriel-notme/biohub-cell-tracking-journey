"""Score every fork of P11 outputs with the frozen b1 fork head and label it TP / FP / U with the official division scorer.
usage: fork_b1.py <gpu> <set,set,...>   (writes /workspace/cl/forkb1/<set>__<movie>.json)"""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
gpu = sys.argv[1]; os.environ['CUDA_VISIBLE_DEVICES'] = gpu
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
import numpy as np
OUT = Path('/workspace/cl/forkb1'); OUT.mkdir(exist_ok=True)


def main():
    from refine_events import EventRefiner
    from cell_event import SCALE, chain, fork_geometry
    import evalx
    from tracking_cellmot.division_metrics import score_divisions
    er = EventRefiner([Path(ART) / 'b1_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, '/workspace/cl/forkb1_cache_%s' % gpu)
    for s in sys.argv[2].split(','):
        for f in sorted(glob.glob('/workspace/cl/ps_p11_%s/graphs/*.json' % s)):
            name = Path(f).stem; of = OUT / ('%s__%s.json' % (s, name))
            if of.exists(): continue
            nodes, edges = evalx.load_graph_json(f)
            out, prev = defaultdict(list), {}
            for e in edges:
                a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a
            forks = [n for n in out if len(out[n]) == 2]
            rows = []
            if forks:
                pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
                trip = [(p, out[p][0], out[p][1]) for p in forks]
                need = {x for t in trip for x in t}
                ids, emb = er.embeddings(name, {n: nodes[n] for n in need}); lookup = {n: i for i, n in enumerate(ids)}
                fg = [fork_geometry(chain(p, prev, pos), chain(a, out, pos), chain(b, out, pos)) for p, a, b in trip]
                fp = er.score('fork', trip, fg, emb, lookup)
                gt, _ = evalx.load_gt(name)
                pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
                res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
                tp = {inv[int(x)] for x in res.tp_forks}; fpk = {inv[int(x)] for x in res.fp_forks}
                eattr = {(int(e['source_id']), int(e['target_id'])): e for e in edges}
                for (p, a, b), pr in zip(trip, fp):
                    prov = 'dc' if (eattr[(p, a)].get('div_complete') or eattr[(p, b)].get('div_complete')) else 'b5'
                    rows.append(dict(p=p, a=a, b=b, t=int(nodes[p]['t']), b1=float(pr), lab='TP' if p in tp else ('FP' if p in fpk else 'U'), prov=prov))
            of.write_text(json.dumps(rows))
            for q in Path('/workspace/cl/forkb1_cache_%s' % gpu).glob('*%s*' % name): q.unlink()
    print('DONE', gpu, flush=True)


if __name__ == '__main__':
    main()
