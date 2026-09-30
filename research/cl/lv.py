"""Leak-free rebuild of LOEO edge-candidate tables (verifier copy; writes only to /workspace/verify2/loeo/lo).
variant 'xrl' : edge cands of movie m (embryo X) built on relink(G0, model trained on embryo X only = relink_base_cross_<other(X)>)
variant 'norl': edge cands built on G0 without relink (same recipe as deployed edge model, which was trained on P3 graphs w/o relink)
variant 'dep' : reproduction of original (DEP relink) for sanity -> must equal /workspace/cl/lo/edge.pkl
Cross edge model for E is then trained ONLY on rows of E' movies whose relink model never saw E."""
import os, sys, json, pickle
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from multiprocessing import Pool
import numpy as np
OUT = Path('/workspace/verify2/loeo/lo'); OUT.mkdir(exist_ok=True, parents=True)
SRC = Path('/workspace/cl/lo')
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
LISTS = {'hold36': 'hold36', 'prev4': 'preview4', 'audit32': 'audit32', 't127a': 't127a', 't127b': 't127b'}
DEP = {'relink': '/workspace/p56stage/relink_lgb.json', 'edge': '/workspace/p56stage/edge_lgb.json'}
PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=16)
OTHER = {'44b6': '6bba', '6bba': '44b6'}
NPROC = int(os.environ.get('NPROC', '22'))


def movies():
    only = os.environ.get('SETS_ONLY')
    return [(s, m) for s, l in LISTS.items() if not only or s in only.split(',') for m in [x.strip() for x in open('/workspace/%s.txt' % l) if x.strip()]]


def g0(s, m):
    import evalx
    return evalx.load_graph_json('/workspace/cl/ps_pre_%s/graphs/%s.json' % (s, m))


def relink_model_for_cands(variant, m):
    if variant == 'dep': return DEP['relink']
    if variant == 'norl': return None
    if variant == 'xrl': return str(SRC / ('relink_base_cross_%s.json' % OTHER[m[:4]]))  # trained on embryo m[:4] only
    raise ValueError(variant)


def ec_job(a):
    s, m, variant = a
    import relink, edge_link
    from edge_cands import label
    nodes, edges = g0(s, m)
    rmod = relink_model_for_cands(variant, m)
    if rmod is not None:
        nodes, edges, _ = relink.apply(nodes, edges, edge_link.load_full(Path(FULL[s]) / (m + '.geff')), rmod, th=0.65)
    rows = edge_link.features(nodes, edges, edge_link.load_full(Path(FULL[s]) / (m + '.geff')))
    label(m, nodes, edges, rows)
    X = np.array([[r.get(k, -1) for k in edge_link.FEATS] for r in rows], np.float32).reshape(-1, len(edge_link.FEATS))
    lab = np.array([1 if r['lab'] == 'P' else (0 if r['lab'] == 'N' else -1) for r in rows])
    return s, m, X, lab


def train_cross(variant):
    import lightgbm as lgb
    res = pickle.load(open(OUT / ('edge_%s.pkl' % variant), 'rb'))
    out = {}
    for e in ['44b6', '6bba']:
        sel = [r for r in res if r[1][:4] != e]
        assert all(r[1][:4] == OTHER[e] for r in sel)
        X = np.concatenate([r[2] for r in sel]); y = np.concatenate([r[3] for r in sel]); k = y >= 0
        b = lgb.train(PARAMS, lgb.Dataset(X[k], y[k]), 400)
        p = OUT / ('edge_%s_cross_%s.json' % (variant, e)); json.dump(b.dump_model(), open(p, 'w')); out[e] = str(p)
        print(variant, 'cross', e, 'train rows', int(k.sum()), 'pos', int((y[k] == 1).sum()), flush=True)
    return out


def ev_job(a):
    s, m, rmode, emodel = a
    import evalx, relink, edge_link, prune
    full = Path(FULL[s]) / (m + '.geff')
    nodes, edges = g0(s, m)
    rmod = {'cross': str(SRC / ('relink_base_cross_%s.json' % m[:4])), 'dep': DEP['relink'], 'none': None}[rmode]
    if rmod is not None: nodes, edges, _ = relink.apply(nodes, edges, edge_link.load_full(full), rmod, th=0.65)
    if emodel is not None: nodes, edges, _ = edge_link.apply(nodes, edges, full, emodel, th=0.4, allow_gap2=True)
    nodes, edges, _ = prune.prune_fragments(nodes, edges, 2)
    r = evalx.score_movie(m, nodes, edges); r['set'] = s; r['rmode'] = rmode; r['emodel'] = emodel
    return r


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'ecands':
        variant = sys.argv[2]
        with Pool(NPROC) as p: res = p.map(ec_job, [(s, m, variant) for s, m in movies()])
        pickle.dump(res, open(OUT / ('edge_%s.pkl' % variant), 'wb'))
        print(variant, 'edge rows', sum(len(r[3]) for r in res), 'labelled', sum((r[3] >= 0).sum() for r in res), 'pos', sum((r[3] == 1).sum() for r in res), flush=True)
    elif cmd == 'train':
        train_cross(sys.argv[2])
    elif cmd == 'eval':
        # arms: name=rmode:emodel ; rmode in {cross,dep,none}; emodel in {orig (cl/lo cross edge), dep, none, <variant>}
        jobs = []; names = []
        for spec in sys.argv[2].split(','):
            name, rest = spec.split('='); rmode, ev = rest.split(':')
            for s, m in movies():
                if ev == 'none': em = None
                elif ev == 'dep': em = DEP['edge']
                elif ev == 'orig': em = str(SRC / ('edge_base_cross_%s.json' % m[:4]))
                else: em = str(OUT / ('edge_%s_cross_%s.json' % (ev, m[:4])))
                jobs.append((s, m, rmode, em)); names.append(name)
        with Pool(NPROC) as p: R = p.map(ev_job, jobs)
        for r, n in zip(R, names): r['arm'] = n
        json.dump(R, open(OUT / ('eval_%s.json' % sys.argv[3]), 'w'))
        print('saved', len(R))
