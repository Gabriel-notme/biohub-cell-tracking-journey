"""r2_caps3 (diagnostic, reads GT): P14 edge labels (TP / evaluable FP / U) by provenance flag x length bin x embryo, and
fork-edge / motion-pass breakdown; plus per-frame node count max (B5 motion-relink 2600 cap). Writes r2_caps3_out.json"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
for p in ['/workspace/cl', '/workspace/cl/ideas', '/workspace/official/src', '/workspace/code', '/workspace/p56stage']:
    sys.path.insert(0, p)
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, .40625, .40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}


def dbin(d):
    return '<=6' if d <= 6 else ('6-10' if d <= 10 else ('10-14' if d <= 14 else '>14'))


def job(f):
    import evalx, combo14
    import tracksdata as td
    from tracking_cellmot import metrics as M
    K = td.DEFAULT_ATTR_KEYS
    name = Path(f).stem; st = f.split('ps_p13_')[1].split('/')[0]; emb = name[:4]
    nodes, edges = evalx.load_graph_json(f)
    nodes, edges, _ = combo14.apply(nodes, edges, fullgeff=FULL[st] + '/' + name + '.geff')
    gt, _ = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    M._evaluate(pred, gt, 'jaccard', evalx.SCALE, 7.)
    ea = M._evaluate_matched_graph(pred, gt)
    lab = {}
    for s, d, m, v in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list(), ea[K.MATCHED_EDGE_MASK].to_list(), ea['pred_valid'].to_list()):
        lab[(inv[s], inv[d])] = 'TP' if m else ('FP' if v else 'U')
    out = defaultdict(list)
    for e in edges: out[int(e['source_id'])].append(int(e['target_id']))
    c = Counter()
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id'])
        fl = ('ll' if 'long_link' in e else 'el' if 'edge_link' in e else 'rl' if 'relink' in e else 'dc' if 'div_complete' in e else
              'dj' if 'dup_join' in e else 'gc' if 'gap_closed' in e else 'g2' if 'gap2_recovered' in e else
              ('mr_' + str(e.get('motion_pass'))) if 'motion_relinked' in e else 'oth')
        if len(out[a]) >= 2: fl += '_fork'
        pa = np.array([max(0, int(round(nodes[a][k]))) for k in 'zyx']) * S; pb = np.array([max(0, int(round(nodes[b][k]))) for k in 'zyx']) * S
        c[(emb, fl, dbin(float(np.linalg.norm(pa - pb))), lab[(a, b)])] += 1
    fr = Counter(int(v['t']) for v in nodes.values())
    return dict(m=name, c=[list(k) + [v] for k, v in c.items()], maxframe=max(fr.values()))


if __name__ == '__main__':
    files = sorted(glob.glob('/workspace/cl/ps_p13_*/graphs/*.json'))
    with Pool(40, maxtasksperchild=4) as p: R = p.map(job, files, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/r2_caps3_out.json', 'w'))
    print('max nodes per frame', max(r['maxframe'] for r in R))
    c = Counter()
    for r in R:
        for k in r['c']: c[tuple(k[:-1])] += k[-1]
    keys = sorted({(k[1], k[2]) for k in c})
    for fl, db in keys:
        s = '%-14s %-6s' % (fl, db)
        for e in ['44b6', '6bba']:
            tp, fp, u = c[(e, fl, db, 'TP')], c[(e, fl, db, 'FP')], c[(e, fl, db, 'U')]
            s += ' | %s TP %6d FP %5d U %6d fpr %.2f' % (e, tp, fp, u, fp / max(1, tp + fp))
        print(s)
