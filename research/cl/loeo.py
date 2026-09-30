"""Leave-one-embryo-out (LOEO) validation of the learned linking steps (relink + free-end edge linking) on all 199 movies.
The hidden test set is embryo-disjoint, so the relevant question is how much of the linking gain survives on an unseen embryo.

G0 = P11 graph before any linking/pruning (ps_pre_<set>/graphs). Pipeline per movie: G0 -> relink(th) -> edge_link(th, gap2) -> post_prune(2).
Model modes:  dep   = deployed relink_lgb.json / edge_lgb.json (trained on t127a+t127b+audit32, both embryos)
              cross = trained only on the OTHER embryo's movies (all sets)       <- test-like
              in    = trained on the SAME embryo, movie-grouped 5-fold (the movie itself never in training)
              none  = no linking (G0 + post_prune)
usage: loeo.py rcands | ecands | train [variant] | eval <variant> <arms,>"""
import os, sys, json, glob, pickle
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
LO = Path('/workspace/cl/lo'); LO.mkdir(exist_ok=True)
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
LISTS = {'hold36': 'hold36', 'prev4': 'preview4', 'audit32': 'audit32', 't127a': 't127a', 't127b': 't127b'}
DEP = {'relink': '/workspace/p56stage/relink_lgb.json', 'edge': '/workspace/p56stage/edge_lgb.json'}
PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=16)


def movies():
    out = []
    only = os.environ.get('SETS_ONLY')
    for s, l in LISTS.items():
        if only and s not in only.split(','): continue
        for m in [x.strip() for x in open('/workspace/%s.txt' % l) if x.strip()]: out.append((s, m))
    return out


def gt_maps(name, nodes, edges):
    import evalx
    from tracking_cellmot.metrics import evaluate
    K = evalx.K
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y)); gp[int(y)] = int(x)
    return p2g, gs, gp


def g0(s, m):
    import evalx
    return evalx.load_graph_json('/workspace/cl/ps_pre_%s/graphs/%s.json' % (s, m))


def rc_job(a):
    s, m = a
    import relink, edge_link
    nodes, edges = g0(s, m)
    rows = relink.features(nodes, edges, edge_link.load_full(Path(FULL[s]) / (m + '.geff')))
    p2g, gs, gp = gt_maps(m, nodes, edges)
    X = np.array([[r.get(k, -1) if r.get(k) is not None else -1 for k in relink.FEATS] for r in rows], np.float32).reshape(-1, len(relink.FEATS))
    lab = []
    for r in rows:
        a_, d_ = p2g.get(r['s']), p2g.get(r['d'])
        valid = (a_ is not None and len(gs.get(a_, [])) > 0) or (d_ is not None and d_ in gp)
        ok = a_ is not None and d_ is not None and d_ in gs.get(a_, [])
        lab.append(1 if ok else (0 if valid else -1))
    return s, m, X, np.array(lab)


def g1(s, m, model):
    import relink, edge_link
    nodes, edges = g0(s, m)
    if model is None: return nodes, edges
    n1, e1, _ = relink.apply(nodes, edges, edge_link.load_full(Path(FULL[s]) / (m + '.geff')), model, th=0.65)
    return n1, e1


def ec_job(a):
    s, m = a
    import edge_link
    sys.path.insert(0, '/workspace/cl')
    from edge_cands import label
    nodes, edges = g1(s, m, DEP['relink'])
    rows = edge_link.features(nodes, edges, edge_link.load_full(Path(FULL[s]) / (m + '.geff')))
    label(m, nodes, edges, rows)
    X = np.array([[r.get(k, -1) for k in edge_link.FEATS] for r in rows], np.float32).reshape(-1, len(edge_link.FEATS))
    lab = np.array([1 if r['lab'] == 'P' else (0 if r['lab'] == 'N' else -1) for r in rows])
    return s, m, X, lab


def dump(kind, res):
    pickle.dump(res, open(LO / ('%s.pkl' % kind), 'wb'))


def train(kind, variant='base', feats_keep=None):
    import lightgbm as lgb
    res = pickle.load(open(LO / ('%s.pkl' % kind), 'rb'))
    emb = lambda m: m[:4]
    models = {}
    def fit(sel):
        X = np.concatenate([r[2] for r in sel]); y = np.concatenate([r[3] for r in sel]); k = y >= 0
        X = X[k]; y = y[k]
        if feats_keep is not None: X = X[:, feats_keep]
        b = lgb.train(PARAMS, lgb.Dataset(X, y), 400)
        return b
    out = {}
    for e in ['44b6', '6bba']:
        other = [r for r in res if emb(r[1]) != e]
        b = fit(other); p = LO / ('%s_%s_cross_%s.json' % (kind, variant, e)); json.dump(b.dump_model(), open(p, 'w')); out[('cross', e)] = str(p)
        same = sorted({r[1] for r in res if emb(r[1]) == e}); rng = np.random.default_rng(0); rng.shuffle(same); fold = {mm: i % 5 for i, mm in enumerate(same)}
        for k in range(5):
            b = fit([r for r in res if emb(r[1]) == e and fold[r[1]] != k]); p = LO / ('%s_%s_in_%s_%d.json' % (kind, variant, e, k)); json.dump(b.dump_model(), open(p, 'w'))
            for mm in same:
                if fold[mm] == k: out[('in', mm)] = str(p)
    json.dump({'%s|%s' % k: v for k, v in out.items()}, open(LO / ('%s_%s_models.json' % (kind, variant)), 'w'))
    return out


def model_for(kind, variant, arm, m):
    if arm == 'dep': return DEP[kind]
    if arm == 'none': return None
    mp = json.load(open(LO / ('%s_%s_models.json' % (kind, variant))))
    return mp['cross|%s' % m[:4]] if arm == 'cross' else mp['in|%s' % m]


def ev_job(a):
    s, m, variant, arm, th_r, th_e = a
    import evalx, relink, edge_link, prune
    full = Path(FULL[s]) / (m + '.geff')
    nodes, edges = g0(s, m)
    rm = model_for('relink', variant, arm, m); em = model_for('edge', variant, arm, m)
    if rm is not None:
        nodes, edges, _ = relink.apply(nodes, edges, edge_link.load_full(full), rm, th=th_r)
    if em is not None:
        nodes, edges, _ = edge_link.apply(nodes, edges, full, em, th=th_e, allow_gap2=os.environ.get('NOGAP2') is None)
    nodes, edges, _ = prune.prune_fragments(nodes, edges, 2)
    r = evalx.score_movie(m, nodes, edges); r['set'] = s; r['arm'] = arm; r['variant'] = variant; r['th'] = (th_r, th_e)
    return r


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'rcands':
        with Pool(64) as p: res = p.map(rc_job, movies())
        dump('relink', res); print('relink rows', sum(len(r[3]) for r in res), 'labelled', sum((r[3] >= 0).sum() for r in res), 'pos', sum((r[3] == 1).sum() for r in res))
    elif cmd == 'ecands':
        with Pool(64) as p: res = p.map(ec_job, movies())
        dump('edge', res); print('edge rows', sum(len(r[3]) for r in res), 'labelled', sum((r[3] >= 0).sum() for r in res), 'pos', sum((r[3] == 1).sum() for r in res))
    elif cmd == 'train':
        variant = sys.argv[2] if len(sys.argv) > 2 else 'base'
        for kind in ['relink', 'edge']: train(kind, variant); print('trained', kind, variant)
    elif cmd == 'eval':
        variant = sys.argv[2]; arms = sys.argv[3].split(','); th_r = float(sys.argv[4]) if len(sys.argv) > 4 else 0.65; th_e = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
        jobs = [(s, m, variant, arm, th_r, th_e) for arm in arms for s, m in movies()]
        with Pool(96) as p: R = p.map(ev_job, jobs)
        tag = '%s_%s_%g_%g%s' % (variant, '-'.join(arms), th_r, th_e, '_nogap2' if os.environ.get('NOGAP2') else '')
        json.dump(R, open(LO / ('eval_%s.json' % tag), 'w'))
        from tracking_cellmot.metrics import summarise
        for arm in arms:
            line = '%-6s %-5s' % (variant, arm)
            for e in ['44b6', '6bba', 'all']:
                rows = [r for r in R if r['arm'] == arm and (e == 'all' or r['movie'].startswith(e))]
                line += ' | %s %.6f' % (e, summarise(rows)['score'])
            print(line, flush=True)
