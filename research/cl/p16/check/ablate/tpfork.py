import sys, json
sys.argv_ = sys.argv
import detail as DT
import evalx, numpy as np
from collections import defaultdict
src, s, name = sys.argv[1], sys.argv[2], sys.argv[3]
paths = {'b5': {'audit32': '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', 't127b': '/workspace/sync3/runs/b5f_t127b/working/lineage_graphs'},
         'p13': {k: '/workspace/cl/ps_p13_%s/graphs' % k for k in ['audit32', 't127b']}, 'p15': {k: '/workspace/cl/ps_p15_%s/graphs' % k for k in ['audit32', 't127b']}}
n, e = evalx.load_graph_json(paths[src][s] + '/%s.json' % name)
gt, _ = evalx.load_gt(name)
d = DT.score_detail(n, e, gt)
out = defaultdict(list)
for x in e: out[int(x['source_id'])].append(int(x['target_id']))
def chain(c, k=8):
    r = []
    while True:
        r.append(c)
        if len(out.get(c, [])) != 1 or len(r) >= k: break
        c = out[c][0]
    return r
print(src, name, 'tpf', d['tpf'], 'fpf', d['fpf'])
for p in d['tpf']:
    print(' TP fork', p, 't', n[p]['t'], 'zyx', [round(float(n[p][q]), 1) for q in 'zyx'], 'g', d['p2g'].get(p))
    for c in out[p]:
        ch = chain(c); print('   child', c, 'len', len(ch), [(x, n[x]['t'], d['p2g'].get(x)) for x in ch], 'next', out.get(ch[-1]))
for r in DT.sb_record(n, e):
    if r['parent'] in d['tpf'] or r['parent'] in d['fpf']: print(' SB hits evaluable fork', r, 'TP' if r['parent'] in d['tpf'] else 'FP')
