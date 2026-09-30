"""Scoring time per movie for 200 candidates (top-100 per type by b1dep from a real dc pool log), v1 vs v2 weights, via the drop-in
divnet_core_v2.DivNetScorer (cold = first call incl. frame loading; warm = second call on the same movie). usage: time_scorer.py <v2 weights glob>"""
import sys, glob, json, time, os
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, '/workspace/cl/p16/divnet2'); sys.path.insert(0, '/workspace/p17ds')
import numpy as np, torch
import divnet_core_v2 as C2, divnet_core as C1
movies = ['44b6_144b256d', '6bba_07477033', '44b6_267148e4']
def triples_and_nodes(name):
    f = glob.glob('/workspace/cl/p16/divnet/ps_dnA_*/dnlog/%s_1.json' % name)[0]; st = f.split('ps_dnA_')[1].split('/')[0]
    R = json.load(open(f)); tr = []
    for ty in ('start', 'stolen'):
        rr = sorted([r for r in R if r['typ'] == ty], key=lambda r: -r['fork_b1'])[:100]; tr += [(r['p'], r['a'], r['b']) for r in rr]
    g = json.load(open('/workspace/cl/ps_p15_%s/graphs/%s.json' % (st, name))); nodes = {int(k): v for k, v in g['nodes'].items()}
    tr = [t for t in tr if all(x in nodes for x in t)]
    return tr, nodes
for lab, mod, ws in [('v1', C1, sorted(glob.glob('/workspace/p17ds/dn1_full_trall_s*.pt'))), ('v2', C2, sorted(glob.glob(sys.argv[1])))]:
    sc = mod.DivNetScorer(ws)
    for name in movies:
        fs = glob.glob('/workspace/cl/p16/divnet/ps_dnA_*/dnlog/%s_1.json' % name)
        if not fs: continue
        tr, nodes = triples_and_nodes(name); z = '/workspace/data/train/%s.zarr' % name
        sc._fc = (None, None); torch.cuda.synchronize(); t0 = time.time(); s = sc.score(z, nodes, tr); torch.cuda.synchronize(); t1 = time.time()
        s2 = sc.score(z, nodes, tr); torch.cuda.synchronize(); t2 = time.time()
        print('%s %s n=%d models=%d cold %.2fs warm %.2fs mean %.3f' % (lab, name, len(tr), len(ws), t1 - t0, t2 - t1, float(np.mean(s))), flush=True)
