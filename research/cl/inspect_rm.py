import os, sys, json, random
os.environ['POLARS_MAX_THREADS'] = '1'
sys.path.insert(0, '/workspace/cl'); sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code')
import warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
import evalx
S = np.array([1.625, 0.40625, 0.40625])
C = json.load(open('/workspace/cl/p16/p19/v_dup/check_rows.json'))


def rp(v):
    return np.array([max(0, int(round(float(v[k])))) for k in 'zyx']) * S


random.seed(0)
stats = defaultdict(list)
for r in C:
    tot = sum(len(v) for v in r['rm_list'].values())
    if not tot:
        continue
    f = '/workspace/cl/p16/ps_p17_%s/graphs/%s.json' % (r['set'], r['movie'])
    nodes, edges = evalx.load_graph_json(f)
    out = defaultdict(list); par = {}
    for e in edges:
        a, b = int(e['source_id']), int(e['target_id']); out[a].append(b); par[b] = a
    byt = defaultdict(list)
    for n, v in nodes.items():
        byt[int(v['t'])].append(n)
    trees = {}

    def nn_other(n, excl):
        t = int(nodes[n]['t'])
        if t not in trees:
            ns = byt[t]; trees[t] = (ns, cKDTree(np.stack([rp(nodes[x]) for x in ns])))
        ns, tr = trees[t]
        d, j = tr.query(rp(nodes[n]), k=min(8, len(ns)))
        for dd, jj in zip(np.atleast_1d(d), np.atleast_1d(j)):
            if ns[jj] != n and ns[jj] not in excl:
                return float(dd)
        return 99.
    gone = set(n for v in r['rm_list'].values() for n in v)
    for kind in ['st', 'tt', 'par']:
        cur = set(r['rm_list'][kind])
        seen = set()
        for n in sorted(cur, key=lambda x: int(nodes[x]['t'])):
            if n in seen:
                continue
            ch = [n]; seen.add(n)
            while len(out.get(ch[-1], [])) == 1 and out[ch[-1]][0] in cur:
                ch.append(out[ch[-1]][0]); seen.add(ch[-1])
            d_head = [nn_other(x, gone - {x}) for x in ch]
            # the surviving remainder of the same track: st -> child of last removed; tt -> parent of first removed;
            # par -> whichever side remains
            if kind == 'st':
                surv = out[ch[-1]][0] if len(out.get(ch[-1], [])) == 1 else None; fwd = True
            elif kind == 'tt':
                surv = par.get(ch[0]); fwd = False
            else:
                if ch[0] in par:
                    surv = par[ch[0]]; fwd = False
                else:
                    surv = out[ch[-1]][0] if len(out.get(ch[-1], [])) == 1 else None; fwd = True
            follow = []
            x = surv
            for k in range(8):
                if x is None or x in gone:
                    break
                follow.append(nn_other(x, gone))
                if fwd:
                    x = out[x][0] if len(out.get(x, [])) == 1 else None
                else:
                    x = par.get(x)
            gd = r['dist']
            stats[kind].append(dict(movie=r['movie'], k=len(ch), t=int(nodes[ch[0]]['t']), d_rm=[round(v, 1) for v in d_head],
                                    surv_nn=[round(v, 1) for v in follow], gt=[gd.get(str(x)) for x in ch],
                                    flags=sorted({k for x in ch for e in edges if int(e['target_id']) == x or int(e['source_id']) == x
                                                  for k in e if k not in ('source_id', 'target_id')})[:6]))
for kind in ['st', 'tt', 'par']:
    E = stats[kind]
    ks = [e['k'] for e in E]
    dr = [v for e in E for v in e['d_rm']]
    sep = [max(e['surv_nn'][:6]) if e['surv_nn'] else None for e in E]
    sepv = np.array([v for v in sep if v is not None])
    print('== %s: events %d nodes %d | chain len hist %s | removed-node NN dist median %.2f p90 %.2f max %.2f' % (
        kind, len(E), sum(ks), np.bincount(ks).tolist(), np.median(dr), np.quantile(dr, .9), max(dr)))
    if len(sepv):
        print('   surviving remainder, next<=6 nodes, max NN dist to another node: <=3.5um %d, 3.5-6 %d, >6 %d, none %d' % (
            (sepv <= 3.5).sum(), ((sepv > 3.5) & (sepv <= 6)).sum(), (sepv > 6).sum(), sum(1 for v in sep if v is None)))
    near = [e for e in E if any(g is not None and g <= 7 for g in e['gt'])]
    print('   events with a removed node <=7um of a GT node: %d' % len(near))
    for e in random.sample(E, min(6, len(E))):
        print('   ', e)
    for e in near[:4]:
        print('   nearGT', e)
