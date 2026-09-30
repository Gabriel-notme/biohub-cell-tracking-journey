"""check2/reimpl: full P-stage rerun graphs (p_stage12.py + cfg_p19r_chk.json) vs post-hoc rule graphs (p19r.apply on the P15 graphs),
exact comparison (node ids, raw float coordinates, all node attributes, edge pairs and edge attribute dicts), plus the other session's
/workspace/cl/p16/ps_p19r_<set> run and the P15 graphs themselves.
usage: cmp_rerun.py <tag> <sets comma> [posthoc module=p19r]"""
import os, sys, json, glob
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[_k] = '1'
sys.path[:0] = ['/workspace/cl/ideas', '/workspace/cl/p16/deploy', '/workspace/code']
from pathlib import Path
from multiprocessing import Pool
from collections import Counter

HERE = '/workspace/cl/p16/check2/reimpl'


def load(f):
    d = json.loads(Path(f).read_text())
    return {int(k): v for k, v in d['nodes'].items()}, d['edges'], d.get('pstage', {})


def ecanon(edges):
    return {(int(e['source_id']), int(e['target_id'])): json.dumps({k: v for k, v in e.items()}, sort_keys=True) for e in edges}


def cmp(A, B):
    na, ea = A[0], A[1]; nb, eb = B[0], B[1]
    oa = set(na) - set(nb); ob = set(nb) - set(na)
    nat = sum(1 for k in set(na) & set(nb) if json.dumps(na[k], sort_keys=True) != json.dumps(nb[k], sort_keys=True))
    EA, EB = ecanon(ea), ecanon(eb)
    eoa = set(EA) - set(EB); eob = set(EB) - set(EA)
    eat = sum(1 for k in set(EA) & set(EB) if EA[k] != EB[k])
    return {'nodes_only_a': len(oa), 'nodes_only_b': len(ob), 'node_attr_diff': nat, 'edges_only_a': len(eoa), 'edges_only_b': len(eob),
            'edge_attr_diff': eat, 'identical': not (oa or ob or nat or eoa or eob or eat)}


def job(a):
    tag, s, f, mod = a
    import importlib
    m = importlib.import_module(mod)
    name = Path(f).stem
    R = load(f)
    P15 = load('/workspace/cl/ps_p15_%s/graphs/%s.json' % (s, name))
    nn, ne, st = m.apply(P15[0], P15[1], name=name, set=s)
    out = {'set': s, 'movie': name, 'rerun_vs_posthoc': cmp(R, (nn, ne)),
           'pstage_keys': {k: v for k, v in R[2].items() if k.startswith('p17') or k.startswith('p19') or 'error' in k},
           'posthoc_st': st, 'n_rerun': len(R[0]), 'n_p15': len(P15[0])}
    o = '/workspace/cl/p16/ps_p19r_%s/graphs/%s.json' % (s, name)
    if os.path.exists(o): out['rerun_vs_other_session'] = cmp(R, load(o))
    return out


if __name__ == '__main__':
    tag, sets = sys.argv[1], sys.argv[2].split(',')
    mod = sys.argv[3] if len(sys.argv) > 3 else 'p19r'
    jobs = [(tag, s, f, mod) for s in sets for f in sorted(glob.glob('%s/ps_%s_%s/graphs/*.json' % (HERE, tag, s)))]
    with Pool(int(os.environ.get('RULE_POOL', '6'))) as p: Rs = p.map(job, jobs, chunksize=1)
    json.dump(Rs, open('%s/cmp_rerun_%s.json' % (HERE, tag), 'w'), indent=0)
    for s in sets:
        rs = [r for r in Rs if r['set'] == s]
        for key in ['rerun_vs_posthoc', 'rerun_vs_other_session']:
            vs = [r[key] for r in rs if key in r]
            if not vs: continue
            tot = Counter()
            for v in vs:
                for k, x in v.items():
                    if k != 'identical': tot[k] += x
            print('%-7s %-24s identical %d/%d %s' % (s, key, sum(v['identical'] for v in vs), len(vs), dict(tot)))
        err = [(r['movie'], {k: v for k, v in r['pstage_keys'].items() if 'error' in k}) for r in rs if any('error' in k for k in r['pstage_keys'])]
        agg = Counter()
        for r in rs:
            for k, v in r['pstage_keys'].items():
                if isinstance(v, (int, float)): agg[k] += v
        print('%-7s movies %d rerun nodes %d P15 nodes %d | pstage p17/p19 stats %s | errors %s' % (s, len(rs), sum(r['n_rerun'] for r in rs),
              sum(r['n_p15'] for r in rs), dict(agg), err))

