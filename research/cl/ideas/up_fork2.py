"""UPSTREAM lens (read-only): P13 forks p->{a,b}; per daughter: fullgraph candidate parent status, whether that candidate parent c is
a P13 track end / has 1 child / is a fork, probs; ILP-prob attribute on the P13 edge; fork label (official)."""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool
import numpy as np
S = np.array([1.625, 0.40625, 0.40625])
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
B5 = {'hold36': '/workspace/runs/b5f_hold36/working/lineage_graphs', 'prev4': '/workspace/runs/b5f_prev4/working/lineage_graphs',
      'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127a': '/workspace/sync4/runs/b5f_t127a/working/lineage_graphs',
      't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'}
SETS = ['hold36', 'prev4', 'audit32', 't127a', 't127b']


def etype(e):
    for k in ['gap_closed', 'edge_link', 'gap2_recovered', 'joint_event', 'relink', 'div_complete', 'safe_division']:
        if k in e: return k
    return e.get('motion_pass', 'other')


def job(a):
    s, f = a
    import evalx, zarr
    from tracking_cellmot.metrics import evaluate
    import tracking_cellmot.division_metrics as DM
    name = Path(f).stem
    nodes, edges = evalx.load_graph_json(f)
    b5n, b5e = evalx.load_graph_json(B5[s] + '/' + name + '.json')
    b5ch = defaultdict(list)
    for e in b5e: b5ch[int(e['source_id'])].append(int(e['target_id']))
    g = zarr.open_group(FULL[s] + '/' + name + '.geff', mode='r')
    fid = np.asarray(g['nodes/ids'][:]).astype(np.int64)
    fE = np.asarray(g['edges/ids'][:]).astype(np.int64); fp = np.asarray(g['edges/props/edge_prob/values'][:])
    fset = set(fid.tolist())
    cpar = {}; cprob = {}; cch = defaultdict(list)
    for (u, v), p in zip(fE.tolist(), fp.tolist()): cpar[v] = u; cprob[(u, v)] = p; cch[u].append(v)
    gt, n_total = evalx.load_gt(name)
    pred, mapping = evalx.to_graph(nodes, edges); inv = {v: k for k, v in mapping.items()}
    res = DM.score_divisions(pred, gt, scale=evalx.SCALE, max_distance=7.)
    tpf = {inv[x] for x in res.tp_forks}; fpf = {inv[x] for x in res.fp_forks}
    ch = defaultdict(list); par = {}; E = {}
    for e in edges:
        u, v = int(e['source_id']), int(e['target_id']); ch[u].append(v); par[v] = u; E[(u, v)] = e
    out = []
    for p, cs in ch.items():
        if len(cs) < 2: continue
        ds = []
        for c in cs:
            e = E[(p, c)]
            if p not in fset or c not in fset: st = 'synth'; cp = None
            elif cpar.get(c) == p: st = 'agree'; cp = p
            elif c in cpar: cp = cpar[c]; st = 'other_in_p13' if cp in nodes else 'other_dropped'
            else: st = 'nocand'; cp = None
            cpn = len(ch.get(cp, [])) if (cp is not None and cp != p and cp in nodes) else None
            ds.append({'st': st, 'et': etype(e), 'ilp': e.get('edge_prob'), 'fp': cprob.get((p, c)), 'cp_nch': cpn,
                       'cp_prob': cprob.get((cp, c)) if cp is not None and cp != p else None, 'in_b5': int(c in b5ch.get(p, []))})
        out.append({'lab': 'TP' if p in tpf else ('FP' if p in fpf else 'U'), 'd': ds, 'b5fork': int(len(b5ch.get(p, [])) >= 2), 'p_ncand': len(cch.get(p, []))})
    return {'set': s, 'movie': name, 'forks': out}


if __name__ == '__main__':
    jobs = [(s, f) for s in SETS for f in sorted(glob.glob('/workspace/cl/ps_p13_%s/graphs/*.json' % s))]
    with Pool(60) as p: R = p.map(job, jobs, chunksize=1)
    json.dump(R, open('/workspace/cl/ideas/up_fork2_rows.json', 'w'))
    def key_fns():
        yield 'b5fork', lambda f: f['b5fork']
        yield 'p_ncand', lambda f: min(f['p_ncand'], 2)
        yield 'n_agree', lambda f: sum(d['st'] == 'agree' for d in f['d'])
        yield 'daughter w/ cand parent that is a P13 END', lambda f: any(d['cp_nch'] == 0 for d in f['d'])
        yield 'daughter w/ cand parent in P13 w/ 1 child', lambda f: any(d['cp_nch'] == 1 for d in f['d'])
        yield 'n ILP-selected daughter edges (ilp>0)', lambda f: sum(1 for d in f['d'] if (d['ilp'] or 0) > 0)
        yield 'min ilp prob bucket', lambda f: min(round(float(d['ilp'] or 0), 1) for d in f['d'])
        yield 'etypes', lambda f: tuple(sorted(d['et'] for d in f['d']))
    for nm, fn in key_fns():
        C = defaultdict(Counter)
        for r in R:
            e = r['movie'][:4]; cl = r['set'] in ('hold36', 'prev4')
            for f in r['forks']:
                k = fn(f); C[k][(e, f['lab'])] += 1
                if cl: C[k][('clean', f['lab'])] += 1
        print('==', nm)
        for k in sorted(C, key=lambda k: str(k)):
            c = C[k]; print('  %-45s' % str(k), ' | '.join('%s TP %3d FP %3d U %4d' % (g, c[(g, 'TP')], c[(g, 'FP')], c[(g, 'U')]) for g in ['44b6', '6bba', 'clean']))
