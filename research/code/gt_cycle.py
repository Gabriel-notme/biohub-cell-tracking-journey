import os, sys, json
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
def one(name):
    from evalx import load_gt, K
    gt, n_total = load_gt(name)
    na = gt.node_attrs(attr_keys=[K.NODE_ID, 't'])
    tt = dict(zip(na[K.NODE_ID].to_list(), na['t'].to_list()))
    ea = gt.edge_attrs(); succ = defaultdict(list); par = {}
    for s, d in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()): succ[s].append(d); par[d] = s
    divs = [n for n in tt if len(succ[n]) >= 2]
    out = []
    for d in divs:
        # walk back to previous division or start
        c = d; age = 0; kind = 'start'
        while c in par:
            c = par[c]; age += 1
            if len(succ[c]) >= 2: kind = 'div'; break
        out.append(dict(movie=name, t=int(tt[d]), age=age, kind=kind, start_t=int(tt[c])))
    return out
if __name__ == '__main__':
    TR = '/workspace/data/train'
    names = sorted(p[:-5] for p in os.listdir(TR) if p.endswith('.geff'))
    with ProcessPoolExecutor(32) as ex: rr = list(ex.map(one, names))
    rows = [r for x in rr for r in x]
    json.dump(rows, open('/workspace/runs/gt_cycle.json', 'w'))
    print('divs', len(rows))
    dd = sorted(r['age'] for r in rows if r['kind'] == 'div')
    print('div-to-div ages (n=%d):' % len(dd), dd)
    ss = sorted((r['age'], r['start_t']) for r in rows if r['kind'] == 'start')
    print('start-to-div ages (n=%d):' % len(ss), ss[:80])
