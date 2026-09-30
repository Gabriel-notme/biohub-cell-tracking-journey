"""Audit: classify B5 edges missing from P13 and P13 edges not in B5, to see what div_complete/dfork/relink/edge_link did.
Diagnostic only (GT is read only to label cases)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}


def gt(name):
    import zarr
    g = zarr.open_group('/workspace/data/train/%s.geff' % name, mode='r')
    ids = np.asarray(g['nodes/ids'][:]).astype(int)
    T = np.asarray(g['nodes/props/t/values'][:]).astype(int)
    V = np.stack([np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'zyx'], 1) * S
    E = np.asarray(g['edges/ids'][:]).astype(int)
    outd = Counter(E[:, 0].tolist())
    return ids, T, V, outd


def job(a):
    s, name = a
    b5 = json.load(open(B5[s] + '/' + name + '.json'))
    p13 = json.load(open('/workspace/cl/ps_p13_%s/graphs/%s.json' % (s, name)))
    nodes = {int(k): v for k, v in p13['nodes'].items()}
    E5 = {(int(e['source_id']), int(e['target_id'])) for e in b5['edges']}
    Ep = {}
    for e in p13['edges']:
        k = 'b5'
        for kk in ('div_complete', 'relink', 'edge_link'):
            if kk in e: k = kk
        Ep[(int(e['source_id']), int(e['target_id']))] = k
    succ5 = defaultdict(list); par5 = {}
    for x, y in E5: succ5[x].append(y); par5[y] = x
    succp = defaultdict(list); parp = {}
    for (x, y), k in Ep.items(): succp[x].append(y); parp[y] = (x, k)
    ids, T, V, outd = gt(name)
    gd = [(int(T[i]), V[i]) for i in range(len(ids)) if outd.get(int(ids[i]), 0) >= 2]
    ga = {}
    for i in range(len(ids)):
        if outd.get(int(ids[i]), 0) >= 1: ga.setdefault(int(T[i]), []).append(V[i])

    def near_div(n):
        t = int(nodes[n]['t']) if n in nodes else None
        if t is None: return None
        p = np.array([nodes[n][k] for k in 'zyx']) * S
        return any(abs(g0 - t) <= 1 and np.linalg.norm(gp - p) <= 7. for g0, gp in gd)

    def annotated(n):
        if n not in nodes: return None
        t = int(nodes[n]['t']); p = np.array([nodes[n][k] for k in 'zyx']) * S
        return any(np.linalg.norm(gp - p) <= 7. for gp in ga.get(t, []))
    cat = Counter(); rec = []
    for (x, y) in E5 - set(Ep):
        if x not in nodes or y not in nodes: cat['pruned_node'] += 1; continue
        py = parp.get(y)
        kids = succp.get(x, [])
        if any(Ep[(x, c)] == 'relink' for c in kids): cat['relink_s'] += 1; continue
        if py is not None and py[1] == 'relink': cat['relink_d'] += 1; continue
        if py is not None and py[1] == 'div_complete': cat['stolen'] += 1; continue
        if len(succ5[x]) == 2:
            c = 'dfork_on_b5fork'
        elif any(Ep[(x, c2)] == 'div_complete' for c2 in kids):
            c = 'dfork_removed_ORIGINAL_of_divcomplete_fork'
        else:
            c = 'stolen_then_dfork_undone' + ('_relinked_' + py[1] if py is not None else '_b_orphan')
            c += '_q_' + ('end' if not kids else 'relinked_' + Ep[(x, kids[0])])
        cat[c] += 1
        rec.append(dict(movie=name, set=s, cat=c, x=x, y=y, t=int(nodes[x]['t']), x_div=near_div(x), x_ann=annotated(x)))
    # div_complete forks surviving
    dcf = [x for x in succp if any(Ep[(x, c)] == 'div_complete' for c in succp[x])]
    cat['dc_edges_surviving'] = len(dcf)
    cat['dc_surviving_as_fork'] = sum(len(succp[x]) == 2 for x in dcf)
    return dict(cat=cat, rec=rec)


if __name__ == '__main__':
    jobs = [(s, os.path.basename(f)[:-5]) for s in B5 for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(48) as p: R = p.map(job, jobs, chunksize=1)
    cat = Counter()
    for r in R: cat.update(r['cat'])
    for k, v in sorted(cat.items()): print(k, v)
    rec = [x for r in R for x in r['rec']]
    c2 = Counter((x['cat'], x['x_div'], x['x_ann']) for x in rec)
    for k, v in sorted(c2.items(), key=str): print(k, v)
    json.dump(rec, open('/workspace/cl/ideas/ca_diff.json', 'w'))
