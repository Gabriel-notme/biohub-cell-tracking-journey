"""Candidates on P5-style graphs + b1 edge-head image probabilities:
  steal: track end s (t) -> node d (t+1) currently linked from single-child parent p (not s)
  swap : two links s->d1, s2->d2 in the same frame, crossed alternative s->d2, s2->d1
usage: steal_swap.py <gpu> <worker> <nworkers>"""
import os, sys, json, time, traceback
gpu, wi, nw = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu); os.environ['OMP_NUM_THREADS'] = '4'; os.environ['POLARS_MAX_THREADS'] = '2'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
SETS = {'t127a': ('/workspace/t127a.txt', '/workspace/cl/p5tr/t127a', '/workspace/sync4/runs/fullgraph_t127a'),
        't127b': ('/workspace/t127b.txt', '/workspace/cl/p5tr/t127b', '/workspace/sync3/runs/fullgraph_t127b'),
        'audit32': ('/workspace/audit32.txt', '/workspace/cl/p5tr/audit32', '/workspace/sync3/runs/fullgraph_audit32'),
        'hold36': ('/workspace/hold36.txt', '/workspace/cl/ps_p5_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/cl/ps_p5_prev4/graphs', '/workspace/runs/fullgraph_prev4')}
OUT = Path('/workspace/cl/ssc'); OUT.mkdir(exist_ok=True)
jobs = [(s, n, g, f) for s, (lst, g, f) in SETS.items() for n in [l.strip() for l in open(lst) if l.strip()]]
jobs = sorted(jobs, key=lambda j: -(Path(j[2]) / (j[1] + '.json')).stat().st_size)[wi::nw]
from refine_events import EventRefiner
from cell_event import SCALE, chain, edge_geometry
import evalx, edge_link, swap
K = evalx.K
er = EventRefiner([Path(ART) / 'b1_best.pt'], '/workspace/data/train', {'fork_ensemble': 'first', 'edge_ensemble': 'last'}, Path('/workspace/cl/ev_cache_ss_%d' % wi))


def _f(v):
    try: return float(v) if v is not None else -1.
    except Exception: return -1.


for s, name, g, f in jobs:
    dst = OUT / ('%s__%s.json' % (s, name))
    if dst.exists(): continue
    t0 = time.time()
    try:
        nodes, edges = evalx.load_graph_json(Path(g) / (name + '.json'))
        full = edge_link.load_full(Path(f) / (name + '.geff'))
        fe = {(int(a), int(b)): float(p) for (a, b), p in zip(full[3].tolist(), full[4].tolist())}
        out = defaultdict(list); prev = {}; eat = {}
        for e in edges:
            a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); prev[b] = a; eat[(a, b)] = e
        pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * SCALE for n, v in nodes.items()}
        frames = defaultdict(list)
        for n, v in nodes.items(): frames[int(v['t'])].append(n)

        def back(n, lim=30):
            k = 0; path = [n]
            while n in prev and k < lim: n = prev[n]; k += 1; path.append(n)
            return k, path

        def fwd(n, lim=30):
            k = 0
            while out.get(n) and k < lim: n = out[n][0]; k += 1
            return k
        steal = []
        for t, ns in frames.items():
            if t + 1 not in frames: continue
            ends = [n for n in ns if not out.get(n)]
            nxt = [d for d in frames[t + 1] if d in prev and len(out.get(prev[d], [])) == 1 and prev[d] in prev and len(out.get(prev[prev[d]], [])) == 1]
            if not ends or not nxt: continue
            tr = cKDTree(np.array([pos[d] for d in nxt]))
            for sn in ends:
                hs, ps = back(sn); vs = pos[ps[0]] - pos[ps[1]] if len(ps) > 1 else np.zeros(3)
                for j in tr.query_ball_point(pos[sn], 10.0):
                    d = nxt[j]; p = prev[d]
                    hp, pp = back(p); vp = pos[pp[0]] - pos[pp[1]] if len(pp) > 1 else np.zeros(3)
                    steal.append(dict(s=sn, d=d, p=p, t=t, d_sd=float(np.linalg.norm(pos[d] - pos[sn])), d_pd=float(np.linalg.norm(pos[d] - pos[p])),
                                      m_sd=float(np.linalg.norm(pos[d] - pos[sn] - vs)), m_pd=float(np.linalg.norm(pos[d] - pos[p] - vp)),
                                      d_sp=float(np.linalg.norm(pos[sn] - pos[p])), hist_s=hs, hist_p=hp, fut_d=fwd(d), fe_sd=fe.get((sn, d), -1.), fe_pd=fe.get((p, d), -1.),
                                      ep_pd=_f(eat[(p, d)].get('edge_prob')), sp_s=float(np.linalg.norm(vs)), sp_p=float(np.linalg.norm(vp)), z=float(pos[sn][0])))
        sw = swap.features(nodes, edges, full)
        pairs = set()
        for r in steal: pairs.add((r['s'], r['d'])); pairs.add((r['p'], r['d']))
        for r in sw: pairs.update([(r['s'], r['d1']), (r['s'], r['d2']), (r['s2'], r['d1']), (r['s2'], r['d2'])])
        pairs = sorted(pairs)
        b1 = {}
        if pairs:
            sub = {n: nodes[n] for pr in pairs for n in pr}
            ids, emb = er.embeddings(name, sub); lookup = {n: i for i, n in enumerate(ids)}
            geo = [edge_geometry(chain(a, prev, pos), chain(b, out, pos)) for a, b in pairs]
            b1 = dict(zip(pairs, er.score('edge', pairs, geo, emb, lookup).tolist()))
        for r in steal: r['b1_sd'] = b1[(r['s'], r['d'])]; r['b1_pd'] = b1[(r['p'], r['d'])]; r['b1_diff'] = r['b1_sd'] - r['b1_pd']
        for r in sw:
            r['b1_c1'] = b1[(r['s'], r['d1'])]; r['b1_c2'] = b1[(r['s2'], r['d2'])]; r['b1_a1'] = b1[(r['s'], r['d2'])]; r['b1_a2'] = b1[(r['s2'], r['d1'])]
            r['b1_diff'] = (r['b1_a1'] + r['b1_a2']) - (r['b1_c1'] + r['b1_c2'])
        # labels
        gt, n_total = evalx.load_gt(name)
        pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
        from tracking_cellmot.metrics import evaluate
        evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
        na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
        p2g = {inv[int(x)]: int(y) for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
        ea = gt.edge_attrs(); gs_ = defaultdict(set); gp_ = {}
        for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs_[int(x)].add(int(y)); gp_[int(y)] = int(x)
        tp = lambda a, b: int(p2g.get(a) is not None and p2g.get(b) is not None and p2g[b] in gs_.get(p2g[a], ()))
        valid = lambda a, b: int((p2g.get(a) is not None and len(gs_.get(p2g[a], ())) > 0) or (p2g.get(b) is not None and p2g[b] in gp_))
        for r in steal:
            g_ = (tp(r['s'], r['d']) - tp(r['p'], r['d'])) - 0.94 * ((valid(r['s'], r['d']) - tp(r['s'], r['d'])) - (valid(r['p'], r['d']) - tp(r['p'], r['d'])))
            r['gain'] = g_; r['lab'] = 'P' if g_ > .01 else ('N' if g_ < -.01 else 'U')
        for r in sw:
            cur = [(r['s'], r['d1']), (r['s2'], r['d2'])]; alt = [(r['s'], r['d2']), (r['s2'], r['d1'])]
            g_ = (sum(tp(*e) for e in alt) - sum(tp(*e) for e in cur)) - 0.94 * (sum(valid(*e) - tp(*e) for e in alt) - sum(valid(*e) - tp(*e) for e in cur))
            r['gain'] = g_; r['lab'] = 'P' if g_ > .01 else ('N' if g_ < -.01 else 'U')
        dst.write_text(json.dumps({'steal': steal, 'swap': sw}))
        for p in Path('/workspace/cl/ev_cache_ss_%d' % wi).glob('*'): p.unlink()
        print('OK', s, name, len(steal), len(sw), round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print('ERR', s, name, repr(e)[:200], traceback.format_exc()[-800:], flush=True)
print('WORKER_DONE', wi, flush=True)
