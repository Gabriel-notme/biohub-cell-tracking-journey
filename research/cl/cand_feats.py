"""Division-completion candidate features on B5 base graphs (b1 fork/edge probs + structure + GT labels).
usage: cand_feats.py <gpu> <worker_idx> <n_workers>"""
import os, sys, json, time, traceback
gpu, wi, nw = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu); os.environ['POLARS_MAX_THREADS'] = '2'; os.environ['OMP_NUM_THREADS'] = '4'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p12ds')
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/b5f_hold36/working/lineage_graphs'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/b5f_prev4/working/lineage_graphs'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs'),
        't127a': ('/workspace/t127a.txt', '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs'),
        't127b': ('/workspace/t127b.txt', '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs')}
OUT = Path('/workspace/cl/cands'); OUT.mkdir(parents=True, exist_ok=True)
jobs = []
for s, (lst, g) in SETS.items():
    for n in [l.strip() for l in open(lst) if l.strip()]: jobs.append((s, n, g))
jobs = sorted(jobs, key=lambda j: -(Path(j[2]) / (j[1] + '.json')).stat().st_size)
mine = jobs[wi::nw]
cfg = json.load(open('/workspace/p12ds/p3_config.json'))
from refine_events import EventRefiner
import div_complete as dc
import evalx
K = evalx.K
er = EventRefiner([Path(ART) / n for n in cfg['models']], '/workspace/data/train', cfg.get('event_config'), Path('/workspace/cl/ev_cache_%d' % wi))
S = np.array([1.625, .40625, .40625])

def gt_info(name, nodes, edges):
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges)
    inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for a, b in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(a)].append(int(b)); gpar[int(b)] = int(a)
    return p2g, gsucc, gpar

for s, name, gdir in mine:
    dst = OUT / ('%s__%s.json' % (s, name))
    if dst.exists(): continue
    t0 = time.time()
    try:
        d = json.load(open(Path(gdir) / (name + '.json')))
        nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
        cands = dc.score_candidates(er, name, nodes, edges, with_edges=True)
        succ = defaultdict(list); par = {}
        for e in edges:
            a_, b_ = int(e['source_id']), int(e['target_id']); succ[a_].append(b_); par[b_] = a_
        pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
        frames = defaultdict(list)
        for n, v in nodes.items(): frames[int(v['t'])].append(n)
        trees = {t: cKDTree(np.array([pos[n] for n in ns])) for t, ns in frames.items()}
        def fwd(n, lim=40):
            k = 0
            while len(succ.get(n, [])) == 1 and k < lim: n = succ[n][0]; k += 1
            return k
        def back(n, lim=40):
            k = 0
            while n in par and k < lim: n = par[n]; k += 1
            return k, n
        ncp = defaultdict(int); ncb = defaultdict(int)
        for c in cands: ncp[c['p']] += 1; ncb[c['b']] += 1
        p2g, gsucc, gpar = gt_info(name, nodes, edges)
        rows = []
        for c in cands:
            p, a, b, q = c['p'], c['a'], c['b'], c['q']
            t = int(nodes[p]['t'])
            hp, _ = back(p)
            pp = par.get(p)
            vp = pos[p] - pos[pp] if pp is not None else np.zeros(3)
            va, vb = pos[a] - pos[p], pos[b] - pos[p]
            cos_ab = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6))
            r = dict(c); r.update(set=s, movie=name, t=t, z=float(pos[p][0]), hist_p=hp, fut_a=fwd(a), fut_b=fwd(b), cos_ab=cos_ab,
                                  d_pa=float(np.linalg.norm(va)), vel_p=float(np.linalg.norm(vp)),
                                  dens_p=len(trees[t].query_ball_point(pos[p], 8.0)), dens_b=len(trees[t + 1].query_ball_point(pos[b], 8.0)),
                                  ncand_p=ncp[p], ncand_b=ncb[b])
            if q is not None:
                hq, qstart = back(q)
                r.update(hist_q=hq, d_qb=float(np.linalg.norm(pos[b] - pos[q])), d_qp=float(np.linalg.norm(pos[q] - pos[p])),
                         qstart_dp=float(np.linalg.norm(pos[qstart] - pos[p])), qstart_dt=int(nodes[p]['t']) - int(nodes[qstart]['t']),
                         q_kids=len(succ.get(q, [])))
            # labels
            gp, ga, gb = p2g.get(p), p2g.get(a), p2g.get(b)
            lab = 'U'
            if gp is not None and len(gsucc.get(gp, [])) >= 1:
                lab = 'N'
                # exact or +-1 frame division
                cand_divs = [gp] + ([gpar[gp]] if gp in gpar else []) + list(gsucc.get(gp, []))
                for gd in cand_divs:
                    ch = gsucc.get(gd, [])
                    if len(ch) != 2: continue
                    lin = [set([x] + gsucc.get(x, [])) for x in ch]
                    ok_a = ga is not None and any(ga in L for L in lin)
                    ok_b = gb is not None and any(gb in L for L in lin)
                    if ok_a and ok_b and not any(ga in L and gb in L for L in lin): lab = 'P'; break
            r['lab'] = lab
            rows.append(r)
        dst.write_text(json.dumps(rows))
        print('OK', s, name, len(rows), sum(r['lab'] == 'P' for r in rows), sum(r['lab'] == 'N' for r in rows), round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print('ERR', s, name, repr(e)[:200], traceback.format_exc()[-800:], flush=True)
print('WORKER_DONE', wi, flush=True)
