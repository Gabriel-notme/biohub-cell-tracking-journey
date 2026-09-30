"""Pseudo-label quality: how precise are B5 forks (as division pseudo-positives), on the annotated part?
For each B5 fork p->(a,b): evaluable if p matches a GT node with children; TP if (as in cand_lab) a GT division at g_p, its parent or child
separates a and b. Reports precision by simple persistence/separation filters, and the count of UNannotated forks (pseudo-label pool).
Writes /workspace/cl/forkq/<set>__<movie>.json rows (p, a, b, t, lab E/T/F, feats)."""
import os, sys, json, glob
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/art_b56/artifact_bundle')
os.environ['BIOHUB_ART'] = '/workspace/art_b56/artifact_bundle'
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
OUT = Path('/workspace/cl/forkq'); OUT.mkdir(exist_ok=True)


def job(args):
    s, f = args
    name = Path(f).stem; of = OUT / ('%s__%s.json' % (s, name))
    if of.exists(): return 1
    import evalx
    from tracking_cellmot.division_metrics import _match_full, _matched_node_attrs
    K = evalx.K
    nodes, edges = evalx.load_graph_json(f)
    gt, _ = evalx.load_gt(name)
    out, prev = defaultdict(list), {}
    for e in edges:
        x, y = int(e['source_id']), int(e['target_id']); out[x].append(y); prev[y] = x
    pos = {n: np.array([v['z'], v['y'], v['x']], np.float32) * evalx.SCALE for n, v in nodes.items()}
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    mp = _match_full(pred, gt, evalx.SCALE, 7.); ma = _matched_node_attrs(mp)
    p2g = {inv[int(a)]: int(b) for a, b in zip(ma[K.NODE_ID].to_list(), ma[K.MATCHED_NODE_ID].to_list())}
    ea = gt.edge_attrs(); gs = defaultdict(list); gp = {}
    for x, y in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): gs[int(x)].append(int(y)); gp[int(y)] = int(x)

    def fwd(n, k=12):
        c = [n]
        while len(c) < k and len(out.get(c[-1], [])) == 1: c.append(out[c[-1]][0])
        return c

    def back(n, k=12):
        c = [n]
        while len(c) < k and c[-1] in prev and len(out.get(prev[c[-1]], [])) == 1: c.append(prev[c[-1]])
        return c
    rows = []
    for p in [n for n in out if len(out[n]) == 2]:
        a, b = out[p]; ca, cb = fwd(a), fwd(b); ph = back(p)
        g_p, g_a, g_b = p2g.get(p), p2g.get(a), p2g.get(b)
        lab = 'U'
        if g_p is not None and gs.get(g_p):
            lab = 'F'
            for gd in [g_p] + ([gp[g_p]] if g_p in gp else []) + list(gs.get(g_p, [])):
                chs = gs.get(gd, [])
                if len(chs) != 2: continue
                lin = [set([x] + gs.get(x, [])) for x in chs]
                if g_a is not None and g_b is not None and any(g_a in L for L in lin) and any(g_b in L for L in lin) and not any(g_a in L and g_b in L for L in lin):
                    lab = 'T'; break
        d = lambda i: float(np.linalg.norm(pos[ca[min(i, len(ca) - 1)]] - pos[cb[min(i, len(cb) - 1)]]))
        rows.append(dict(p=p, a=a, b=b, t=int(nodes[p]['t']), lab=lab, la=len(ca), lb=len(cb), lp=len(ph), d0=d(0), d2=d(2), d4=d(4), d8=d(8),
                         dpa=float(np.linalg.norm(pos[p] - pos[a])), dpb=float(np.linalg.norm(pos[p] - pos[b]))))
    of.write_text(json.dumps(rows))
    return 1


if __name__ == '__main__':
    jobs = [(s, f) for s in B5 for f in sorted(glob.glob(B5[s] + '/*.json'))]
    with Pool(64) as p: R = p.map(job, jobs)
    print('done', sum(R), len(jobs))
