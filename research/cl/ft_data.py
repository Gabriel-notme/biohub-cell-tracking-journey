"""Training/eval data for fine-tuning a division-candidate classifier: div_complete candidates on B5 graphs with GT labels
(positives = P or D, i.e. true division triples; negatives = N). Parent patch = p for 'start', p/q midpoint for 'stolen'.
Keeps all positives and up to 80 negatives per movie (half the hardest by b1 fork prob, half random).
usage: ft_data.py <gpu-unused> <sets> -> /workspace/cl/ft/<set>__<movie>.npz"""
import os, sys, json, glob
ART = '/workspace/art_b56/artifact_bundle'
sys.path.insert(0, ART); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
OUT = Path('/workspace/cl/ft'); OUT.mkdir(exist_ok=True)
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def job(a):
    s, f = a
    from cell_event import SCALE, Movie, chain, fork_geometry
    import evalx
    name = Path(f).stem; of = OUT / ('%s__%s.npz' % (s, name))
    if of.exists(): return 1
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31)
    labs = [r for r in json.load(open(f)) if r['lab'] in ('P', 'D', 'N')]
    dump = Path('/workspace/cl/cands_p8/%s/%s.json' % (s, name))
    base = {(c['p'], c['a'], c['b']): c['fork'] for c in json.load(open(dump))} if dump.exists() else {}
    pos_ = [r for r in labs if r['lab'] != 'N']; neg = [r for r in labs if r['lab'] == 'N']
    neg.sort(key=lambda r: -base.get((r['p'], r['a'], r['b']), 0))
    hard = neg[:40]; rest = neg[40:]
    rnd = [rest[i] for i in rng.choice(len(rest), min(40, len(rest)), replace=False)] if rest else []
    keep = pos_ + hard + rnd
    if not keep: np.savez_compressed(of, n=0); return 1
    nodes, edges = evalx.load_graph_json(Path(B5[s]) / (name + '.json'))
    out, prev = defaultdict(list), {}
    for e in edges:
        a_, b_ = int(e['source_id']), int(e['target_id']); out[a_].append(b_); prev[b_] = a_
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    raw = {n: [v['z'], v['y'], v['x']] for n, v in nodes.items()}
    mv = Movie(Path('/workspace/data/train') / (name + '.zarr'), context=5)
    T, C, G, Y, BASE, TYP = [], [], [], [], [], []
    for r in keep:
        p, a_, b_, q = r['p'], r['a'], r['b'], r['q']
        t = int(nodes[p]['t'])
        if r['typ'] == 'stolen' and q is not None:
            par_c = [(x + y) / 2 for x, y in zip(raw[p], raw[q])]
            hp = chain(p, prev, pos); hq = chain(q, prev, pos); k = min(len(hp), len(hq)); hist = (hp[:k] + hq[:k]) / 2
        else:
            par_c = raw[p]; hist = chain(p, prev, pos)
        T += [t, t + 1, t + 1]; C += [par_c, raw[a_], raw[b_]]
        G.append(fork_geometry(hist, chain(a_, out, pos), chain(b_, out, pos)))
        Y.append(int(r['lab'] != 'N')); BASE.append(base.get((p, a_, b_), -1.0)); TYP.append(int(r['typ'] == 'stolen'))
    X = mv.patches(T, C).reshape(len(keep), 3, 5, 12, 24, 24)
    np.savez_compressed(of, n=len(keep), X=X, G=np.stack(G).astype(np.float32), y=np.array(Y), base=np.array(BASE, np.float32), typ=np.array(TYP),
                        n_neg_total=len(neg), n_neg_kept=len(hard) + len(rnd))
    return 1


if __name__ == '__main__':
    jobs = [(s, f) for s in sys.argv[2].split(',') for f in sorted(glob.glob('/workspace/cl/cl2/%s/*.json' % s))]
    with Pool(48) as p: print('done', sum(p.map(job, jobs)))
