"""Quick check of r3_rlf on a few movies: sanity counts, timing, and agreement of exact=False with p56stage/p15_post.relinefit."""
import sys, json, glob, os, time
import numpy as np
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/p56stage'); sys.path.insert(0, '/workspace/code')
import evalx
from ideas import r3_rlf
import p15_post
FULL = {'hold36': '/workspace/runs/fullgraph_hold36', 'prev4': '/workspace/runs/fullgraph_prev4', 'audit32': '/workspace/sync3/runs/fullgraph_audit32',
        't127a': '/workspace/sync4/runs/fullgraph_t127a', 't127b': '/workspace/sync3/runs/fullgraph_t127b'}
for s in ['hold36', 'audit32', 't127a']:
    for f in sorted(glob.glob('/workspace/cl/ps_p14_%s/graphs/*.json' % s))[:2]:
        name = os.path.basename(f)[:-5]
        nodes, edges = evalx.load_graph_json(f)
        kw = dict(name=name, set=s, fullgeff=FULL[s] + '/' + name + '.geff', zarr=None)
        t0 = time.time(); a, _, sa = r3_rlf.apply(nodes, edges, w=0.5, exact=True, **kw); t1 = time.time()
        b, _, sb = r3_rlf.apply(nodes, edges, w=0.5, exact=False, **kw)
        c, _, sc = p15_post.relinefit(nodes, edges, r3_rlf.REF[s] + '/' + name + '.json', kw['fullgeff'], w=0.5)
        P = lambda g, k: np.array([g[k][q] for q in 'zyx'])
        dbc = max(np.linalg.norm((P(b, k) - P(c, k)) * r3_rlf.S) for k in nodes)
        dab = [np.linalg.norm((P(a, k) - P(b, k)) * r3_rlf.S) for k in nodes]
        print(s, name, '%.2fs' % (t1 - t0), 'exact', sa, '\n   cap3', sb, '\n   p15', sc, '\n   max|cap3-p15| %.2e  exact-vs-cap3: n>0.01um %d  n>0.5um %d max %.2f' % (
            dbc, sum(x > .01 for x in dab), sum(x > .5 for x in dab), max(dab)), flush=True)
