"""chk2 placebo (key: placebo): shared library. Pure functions; writes nothing.
Sequential re-application of the deployed P19-R families on P15, piece typing, and an exact reduced-graph scorer
(official evaluate() on the union of weakly connected pred components that contain a GT-matchable node; num_pred_nodes of the
full graph). Exactness of the reduced scorer is verified against evalx.score_movie in the runs."""
import os, sys, json
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for _p in ['/workspace/p12ds', '/workspace/cl/p16/deploy', '/workspace/cl/ideas', '/workspace/code', '/workspace/official/src', '/workspace/cl']:
    if _p not in sys.path: sys.path.insert(0, _p)
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')
import numpy as np

SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']
B5 = {'hold36': '/workspace/runs/b5f_hold36', 'prev4': '/workspace/runs/b5f_prev4', 'audit32': '/workspace/sync3/runs/b5f_audit32',
      't127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b'}
P15 = '/workspace/cl/ps_p15_%s/graphs'
S = np.array([1.625, .40625, .40625])
FAMS = ['cd', 'ff', 'st', 'tt', 'par', 'bd']


def load(p):
    d = json.load(open(p)); return {int(k): v for k, v in d['nodes'].items()}, d['edges']


def sub(nodes, edges, drop):
    return ({k: v for k, v in nodes.items() if k not in drop},
            [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop])


def topo(edges):
    succ = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); succ[a].append(b); par[b] = a
    return succ, par


def wcc(nodes, edges):
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); adj[a].append(b); adj[b].append(a)
    lab = {}; comps = []
    for n in nodes:
        if n in lab: continue
        cid = len(comps); c = [n]; st = [n]; lab[n] = cid
        while st:
            u = st.pop()
            for v in adj.get(u, ()):
                if v not in lab: lab[v] = cid; st.append(v); c.append(v)
        comps.append(c)
    return lab, comps


def comps_of(nset, edges):
    adj = defaultdict(list)
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        if a in nset and b in nset: adj[a].append(b); adj[b].append(a)
    seen = set(); out = []
    for n in sorted(nset):
        if n in seen: continue
        c = []; st = [n]; seen.add(n)
        while st:
            u = st.pop(); c.append(u)
            for v in adj[u]:
                if v not in seen: seen.add(v); st.append(v)
        out.append(sorted(c))
    return out


def apply_rules(n15, e15, refp, start_r=2.5, par_r=3.5, margin=2.0, bminlen=6):
    """Deployed P19-R, family by family (identical calls/arguments to ideas/p19r.py + p19_dup.post). Returns steps
    [(fam, pre_nodes, pre_edges, removed_set)], final graph, B5 reference fork daughters."""
    import p17_post, p19_dup, p14_post, p19_edge_deploy
    D = p19_dup.ref_fork_daughters(refp)
    steps = []
    a, b = n15, e15
    x, y, _ = p17_post.cutdup(a, b); steps.append(('cd', a, b, set(a) - set(x))); a, b = x, y
    x, y, _ = p17_post.forkfrag(a, b, refp); steps.append(('ff', a, b, set(a) - set(x))); a, b = x, y
    x, y, _ = p19_dup.start_trim(a, b, D, r=start_r, minlen=3, iters=5); steps.append(('st', a, b, set(a) - set(x))); a, b = x, y
    x, y, _ = p14_post.term_trim(a, b, join=None); steps.append(('tt', a, b, set(a) - set(x))); a, b = x, y
    x, y, _, _ = p19_dup.par_dup(a, b, r=par_r, minrun=3, which='shorter'); steps.append(('par', a, b, set(a) - set(x))); a, b = x, y
    r = p19_edge_deploy.yx_border_stubs(a, b, shape_yx=(256, 256), minlen=bminlen, margin=margin)
    x, y = r[0], r[1]; steps.append(('bd', a, b, set(a) - set(x))); a, b = x, y
    return steps, (a, b), D


def ekey(edges):
    return {(int(e['source_id']), int(e['target_id'])) for e in edges}


def piece_types(nodes_pre, edges_pre, rm):
    """Type every weakly connected piece of a removed set in the graph it was removed from."""
    succ, par = topo(edges_pre); lab, comps = wcc(nodes_pre, edges_pre)
    out = []
    for P in comps_of(rm, edges_pre):
        Ps = set(P)
        ext_in = [n for n in P if n in par and par[n] not in Ps]
        ext_out = [c for n in P for c in succ.get(n, []) if c not in Ps]
        whole = len(comps[lab[P[0]]]) == len(P)
        typ = 'whole' if whole else ('head' if not ext_in else ('tail' if not ext_out else 'bridge'))
        forky = any(len(succ.get(x, [])) >= 2 for x in P) or any(len(succ.get(par[n], [])) >= 2 for n in ext_in)
        out.append({'nodes': P, 'size': len(P), 'typ': typ, 'forky': bool(forky), 'nin': len(ext_in), 'nout': len(ext_out)})
    return out


class Scorer:
    """Exact official scoring of deletion variants of one base graph, restricted to GT-relevant components."""

    def __init__(self, name, nodes, edges, rad=7.05):
        import evalx
        from scipy.spatial import cKDTree
        self.name = name; self.nodes = nodes; self.edges = edges; self.N = len(nodes)
        self.gt, self.nt = evalx.load_gt(name)
        ga = self.gt.node_attrs(attr_keys=['t', 'z', 'y', 'x'])
        gpos = defaultdict(list)
        for t, z, y, x in zip(ga['t'].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list()):
            gpos[int(t)].append(np.array([z, y, x], float) * S)
        trees = {t: cKDTree(np.stack(v)) for t, v in gpos.items()}
        self.matchable = set()
        for n, v in nodes.items():
            t = int(v['t'])
            if t not in trees: continue
            p = np.array([max(0, int(round(float(v[k])))) for k in 'zyx'], float) * S
            if trees[t].query_ball_point(p, rad): self.matchable.add(n)
        lab, comps = wcc(nodes, edges)
        good = {lab[n] for n in self.matchable}
        self.R0 = {n for c in good for n in comps[c]}
        self.order = [n for n in nodes if n in self.R0]
        self.eR0 = [e for e in edges if int(e['source_id']) in self.R0]

    def score(self, drop):
        import evalx
        from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
        drop = set(drop)
        nn = {n: self.nodes[n] for n in self.order if n not in drop}
        ee = [e for e in self.eR0 if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
        npred = self.N - len(drop & self.nodes.keys())
        if not nn or not ee:  # degenerate reduced graph: fall back to the full official path
            fn, fe = sub(self.nodes, self.edges, drop)
            r = evalx.score_movie(self.name, fn, fe); r['reduced'] = 0; return r
        pred, mp = evalx.to_graph(nn, ee)
        er = evaluate(pred, self.gt, scale=evalx.SCALE, max_distance=7.)
        er = er._replace(num_pred_nodes=npred)
        row = per_sample_metrics(er, self.nt, node_recall(pred, self.gt)); row['movie'] = self.name; row['n_total'] = self.nt
        row['reduced'] = 1
        return row
