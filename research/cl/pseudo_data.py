"""Semi-supervised division data. Annotation covers ~1-2% of cells, but the images contain many more divisions.
Pseudo-positives: B5 forks (p,a,b) with b1 fork prob >= 0.95 in UNANNOTATED regions (p not matched to GT) whose daughters
both persist >= 5 frames and separate (d_ab at +3 > d_ab at +1). Pseudo-negatives: division-completion candidates with
b1 fork prob <= 0.05 in unannotated regions. GT-labelled candidates (P/D/N) are kept for evaluation only.
Patches: parent (p for B5 forks and start candidates, p/q midpoint for stolen), a, b.
usage: pseudo_data.py <sets> -> /workspace/cl/pl/<set>__<movie>.npz"""
import os, sys, json, glob
ART = '/workspace/art_b56/artifact_bundle'; sys.path.insert(0, ART); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds')
os.environ['BIOHUB_ART'] = ART
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
OUT = Path('/workspace/cl/pl'); OUT.mkdir(exist_ok=True)
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
    nodes, edges = evalx.load_graph_json(Path(B5[s]) / (name + '.json'))
    out, prev = defaultdict(list), {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y] = x
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
    raw = {n: [v['z'], v['y'], v['x']] for n, v in nodes.items()}
    labs = json.load(open(f))  # cl2 candidate rows incl. U
    lab_p = defaultdict(set)
    for r in labs:
        if r['lab'] != 'U': lab_p[r['p']].add(r['lab'])
    # GT-matched parents: any candidate p with a non-U label is in an annotated region; B5 forks there are excluded from pseudo labels
    annotated = set(lab_p)

    def fwd(n, k):
        c = [n]
        while len(c) < k and len(out.get(c[-1], [])) == 1: c.append(out[c[-1]][0])
        return c
    rows = []  # (kind, parent_coord, t, a, b, geometry, label, typ)
    # pseudo-positives from B5 forks
    for p in [n for n in out if len(out[n]) == 2]:
        if p in annotated: continue
        a_, b_ = out[p]
        ca, cb = fwd(a_, 6), fwd(b_, 6)
        if len(ca) < 5 or len(cb) < 5: continue
        if np.linalg.norm(pos[ca[3]] - pos[cb[3]]) <= np.linalg.norm(pos[a_] - pos[b_]): continue
        rows.append(('pseudo_pos', raw[p], int(nodes[p]['t']), a_, b_, fork_geometry(chain(p, prev, pos), chain(a_, out, pos), chain(b_, out, pos)), 1, 2))
    # candidates: GT-labelled (eval) + unlabelled low-score pseudo-negatives
    U = [r for r in labs if r['lab'] == 'U']; L = [r for r in labs if r['lab'] != 'U']
    Lneg = [r for r in L if r['lab'] == 'N']; Lpos = [r for r in L if r['lab'] != 'N']
    Lneg = [Lneg[i] for i in rng.choice(len(Lneg), min(len(Lneg), 120), replace=False)] if Lneg else []
    Usel = [U[i] for i in rng.choice(len(U), min(len(U), 400), replace=False)] if U else []
    for kind, rs in [('gt', Lpos + Lneg), ('unl', Usel)]:
        for r in rs:
            p, a_, b_, q = r['p'], r['a'], r['b'], r['q']
            par = [(x + y) / 2 for x, y in zip(raw[p], raw[q])] if (r['typ'] == 'stolen' and q is not None) else raw[p]
            lab = {'P': 1, 'D': 2, 'N': 0}.get(r['lab'], -1)
            rows.append((kind, par, int(nodes[p]['t']), a_, b_, fork_geometry(chain(p, prev, pos), chain(a_, out, pos), chain(b_, out, pos)), lab, int(r['typ'] == 'stolen')))
    if not rows: np.savez_compressed(of, n=0); return 1
    mv = Movie(Path('/workspace/data/train') / (name + '.zarr'), context=5)
    T, C = [], []
    for kind, par, t, a_, b_, g, lab, typ in rows:
        T += [t, t + 1, t + 1]; C += [par, raw[a_], raw[b_]]
    X = mv.patches(T, C).reshape(len(rows), 3, 5, 12, 24, 24)
    kinds = np.array([{'pseudo_pos': 0, 'gt': 1, 'unl': 2}[r[0]] for r in rows])
    np.savez_compressed(of, n=len(rows), X=X, G=np.stack([r[5] for r in rows]).astype(np.float32), kind=kinds,
                        lab=np.array([r[6] for r in rows]), typ=np.array([r[7] for r in rows]), n_neg_total=len([r for r in L if r['lab'] == 'N']),
                        n_neg_kept=len(Lneg))
    return 1


if __name__ == '__main__':
    jobs = [(s, f) for s in sys.argv[1].split(',') for f in sorted(glob.glob('/workspace/cl/cl2/%s/*.json' % s))]
    with Pool(48) as p: print('done', sum(p.map(job, jobs)))
