"""sens checker driver: rule_eval.job on src p15 with ideas.chk_sens; rows written to this dir (strict format)."""
import os, sys, json, glob, time
for _k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLOSC_NTHREADS']:
    os.environ[_k] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
from multiprocessing import Pool
import numpy as np
import rule_eval

if __name__ == '__main__':
    tag = sys.argv[1]; V = [None] + json.loads(sys.argv[2]); lim = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    jobs = [('ideas.chk_sens', f, V) for s, d in rule_eval.SRC['p15'].items() for f in sorted(glob.glob(d + '/*.json'))]
    if lim: jobs = jobs[:lim]
    t0 = time.time()
    with Pool(int(os.environ.get('RULE_POOL', '6')), maxtasksperchild=4) as p: R = [r for rs in p.map(rule_eval.job, jobs, chunksize=1) for r in rs]
    out = '/workspace/cl/p16/check/sens/rows_%s.json' % tag
    json.dump(R, open(out, 'w')); json.dump(V, open('/workspace/cl/p16/check/sens/variants_%s.json' % tag, 'w'))
    print('wrote', out, 'rows', len(R), 'sec %.0f' % (time.time() - t0), flush=True)
