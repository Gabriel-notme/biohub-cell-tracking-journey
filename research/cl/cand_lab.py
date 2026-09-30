"""Label every div_complete candidate (p,a,b,q,typ) on a set of graphs with GT (P new division / D already recovered / N not a division / U unknown)
and compute structural features incl. q-track anatomy. Output one JSON per movie in /workspace/cl/cl2/<tag>/."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/art_b56/artifact_bundle')
os.environ['BIOHUB_ART'] = '/workspace/art_b56/artifact_bundle'
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np


def feats(nodes, out, prev, pos, p, a, b, q):
    def back(n, k=40):
        c = [n]
        while len(c) < k and c[-1] in prev and len(out.get(prev[c[-1]], [])) == 1: c.append(prev[c[-1]])
        return c

    def fwd(n, k=40):
        c = [n]
        while len(c) < k and len(out.get(c[-1], [])) == 1: c.append(out[c[-1]][0])
        return c
    ph = back(p); af = fwd(a); bf = fwd(b)
    f = dict(p_hist=len(ph), a_fut=len(af), b_fut=len(bf), p_root=int(ph[-1] not in prev), p_after_div=int(ph[-1] in prev))
    f['d_pb'] = float(np.linalg.norm(pos[p] - pos[b])); f['d_ab'] = float(np.linalg.norm(pos[a] - pos[b])); f['d_pa'] = float(np.linalg.norm(pos[p] - pos[a]))
    # divergence of a and b over next frames
    for k in [2, 4]:
        f['d_ab_%d' % k] = float(np.linalg.norm(pos[af[min(k, len(af) - 1)]] - pos[bf[min(k, len(bf) - 1)]]))
    if q is not None:
        qh = back(q)
        f['q_hist'] = len(qh); f['q_root'] = int(qh[-1] not in prev); f['d_qp'] = float(np.linalg.norm(pos[q] - pos[p])); f['d_qb'] = float(np.linalg.norm(pos[q] - pos[b]))
        # p-track node at the time of q-track start
        tq = int(nodes[qh[-1]]['t']); pn = [x for x in ph if int(nodes[x]['t']) == tq]
        f['qstart_dp'] = float(np.linalg.norm(pos[qh[-1]] - pos[pn[0]])) if pn else -1.
        f['qstart_in_phist'] = int(bool(pn))
        dd = [float(np.linalg.norm(pos[x] - pos[y])) for x in qh for y in ph if nodes[x]['t'] == nodes[y]['t']]
        f['qp_min'] = min(dd) if dd else -1.; f['qp_mean'] = float(np.mean(dd)) if dd else -1.
    return f


def job(args):
    tag, f, gtdir = args
    name = Path(f).stem
    od = Path('/workspace/cl/cl2/%s' % tag); od.mkdir(parents=True, exist_ok=True)
    of = od / (name + '.json')
    if of.exists(): return 1
    if not Path('/workspace/data/train/%s.geff/nodes/props' % name).exists(): return 0
    import evalx, div_complete as dc
    from tracking_cellmot.division_metrics import score_divisions, _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    try: gt, _ = evalx.load_gt(name)
    except Exception: return 0
    rows, out, prev, pos = dc.candidates(nodes, edges)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    recovered = {int(d) for d, v in res.scores.items() if v}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y)); gp[int(y)] = int(x)
    outr = []
    for p, a, b, q, typ in rows:
        g_p, g_a, g_b = p2g.get(p), p2g.get(a), p2g.get(b)
        lab = 'U'
        if g_p is not None and gs.get(g_p):
            lab = 'N'
            for gd in [g_p] + ([gp[g_p]] if g_p in gp else []) + list(gs.get(g_p, [])):
                chs = gs.get(gd, [])
                if len(chs) != 2: continue
                lin = [set([x] + gs.get(x, [])) for x in chs]
                if g_a is not None and g_b is not None and any(g_a in L for L in lin) and any(g_b in L for L in lin) and not any(g_a in L and g_b in L for L in lin):
                    lab = 'D' if gd in recovered else 'P'; break
        r = dict(p=p, a=a, b=b, q=q, typ=typ, lab=lab, t=int(nodes[p]['t']), **feats(nodes, out, prev, pos, p, a, b, q))
        outr.append(r)
    of.write_text(json.dumps(outr))
    return 1


if __name__ == '__main__':
    SETS = json.loads(sys.argv[1])
    jobs = [(tag, f, None) for tag, d in SETS.items() for f in sorted(glob.glob(d + '/*.json'))]
    with Pool(64) as p: R = p.map(job, jobs)
    print('done', sum(R), 'of', len(jobs))
