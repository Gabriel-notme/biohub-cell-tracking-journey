"""Runs of consecutive unmatched GT nodes on annotated tracks: are they flanked by a pred track END (before) and a
pred track START (after)? Would linear interpolation between them land within 7um of the GT nodes?"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
G = sys.argv[1] if len(sys.argv) > 1 else '/workspace/cl/ps_p5_hold36/graphs'
LST = sys.argv[2] if len(sys.argv) > 2 else '/workspace/hold36.txt'


def job(name):
    import evalx
    K = evalx.K
    nodes, edges = evalx.load_graph_json(Path(G) / (name + '.json'))
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    from tracking_cellmot.metrics import evaluate
    evaluate(pred, gt, scale=evalx.SCALE, max_distance=7.)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    g2p = {int(y): inv[int(x)] for x, y in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if y is not None and int(y) != -1}
    ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't', 'z', 'y', 'x'])
    gpos = {int(i): np.array([z, y, x]) * S for i, z, y, x in zip(ga[K.NODE_ID].to_list(), ga['z'].to_list(), ga['y'].to_list(), ga['x'].to_list())}
    ea = gt.edge_attrs(); gsucc = defaultdict(list); gpar = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gsucc[int(x)].append(int(y)); gpar[int(y)] = int(x)
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    pos = {n: np.array([v['z'], v['y'], v['x']]) * S for n, v in nodes.items()}
    c = Counter()
    # find maximal runs of unmatched GT nodes along single-successor GT chains
    for g in gpos:
        if g in g2p: continue
        pg = gpar.get(g)
        if pg is None or pg not in g2p: continue  # run must start right after a matched GT node
        run = [g]
        while len(gsucc.get(run[-1], [])) == 1 and gsucc[run[-1]][0] not in g2p and len(run) < 12: run.append(gsucc[run[-1]][0])
        nxt = gsucc.get(run[-1], [])
        if len(nxt) != 1 or nxt[0] not in g2p: c['run_open'] += 1; continue
        L = len(run); s = g2p[pg]; d = g2p[nxt[0]]
        flank = ('s_end' if not succ.get(s) else 's_linked') + '/' + ('d_start' if d not in par else 'd_taken')
        # interpolation accuracy
        ok = 0
        for i, gg in enumerate(run):
            w = (i + 1) / (L + 1); ip = pos[s] * (1 - w) + pos[d] * w
            ok += int(np.linalg.norm(ip - gpos[gg]) <= 7)
        c[('L%d' % min(L, 6), flank)] += 1; c[('L%d' % min(L, 6), flank, 'interp_ok_nodes')] += ok; c[('L%d' % min(L, 6), flank, 'nodes')] += L
    return c


if __name__ == '__main__':
    names = [l.strip() for l in open(LST) if l.strip()]
    with Pool(36) as pool: cs = pool.map(job, names)
    tot = Counter()
    for c in cs: tot.update(c)
    for k, v in sorted(tot.items(), key=lambda kv: str(kv[0])): print(k, v)
