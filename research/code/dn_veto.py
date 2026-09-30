import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
import numpy as np

def compute(gdir, outp):
    import torch
    from divnet_data import Vol
    from divnet_train import DivNet
    import evalx
    nets = []
    for m in ['/workspace/divnet/divnet_s0.pt', '/workspace/divnet/divnet_s1.pt']:
        ck = torch.load(m, map_location='cpu'); n = DivNet(**ck['config']); n.load_state_dict(ck['state_dict']); nets.append(n.cuda().eval())
    res = {}
    for path in sorted(glob.glob(gdir + '/*.json')):
        name = Path(path).stem; nodes, edges = evalx.load_graph_json(path)
        succ = defaultdict(list)
        for e in edges: succ[int(e['source_id'])].append(int(e['target_id']))
        forks = [p for p, k in succ.items() if len(k) == 2]
        if not forks: res[name] = {}; continue
        V = Vol(name)
        X = np.stack([V.crop(int(nodes[p]['t']), nodes[p]['z'], nodes[p]['y'], nodes[p]['x']) for p in forks])
        x = torch.from_numpy(X.astype(np.float32)).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            lg = sum(n(x).float() + n(x.flip(3)).float() + n(x.flip(4)).float() + n(torch.rot90(x, 1, (3, 4))).float() for n in nets) / 8
        res[name] = {int(p): float(v) for p, v in zip(forks, torch.sigmoid(lg).cpu().numpy())}
    json.dump(res, open(outp, 'w'))

def veto(nodes, edges, dn, v):
    succ = defaultdict(list)
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id']))
    rm = set()
    for p, prob in dn.items():
        if prob < v and len(succ.get(p, [])) == 2:
            a, b = succ[p]
            pa = np.array([nodes[a][k] for k in 'zyx']) * [1.625, .40625, .40625]; pb = np.array([nodes[b][k] for k in 'zyx']) * [1.625, .40625, .40625]; pp = np.array([nodes[p][k] for k in 'zyx']) * [1.625, .40625, .40625]
            rm.add((p, b if np.linalg.norm(pb - pp) >= np.linalg.norm(pa - pp) else a))
    return [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm], len(rm)

def job(args):
    gdir, name, dn, vs = args
    import evalx
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    rows = [dict(evalx.score_movie(name, nodes, edges), cfg=-1)]
    for ci, v in enumerate(vs):
        ne, n = veto(nodes, edges, dn, v)
        rows.append(dict(evalx.score_movie(name, nodes, ne), cfg=ci, vetoed=n))
    return rows

if __name__ == '__main__':
    gdir, dnp, vs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    if not os.path.exists(dnp): compute(gdir, dnp)
    dns = json.load(open(dnp))
    with Pool(6) as pool: allrows = [r for rs in pool.map(job, [(gdir, n, {int(k): v for k, v in d.items()}, vs) for n, d in dns.items()]) for r in rs]
    from tracking_cellmot.metrics import summarise
    base = summarise([r for r in allrows if r['cfg'] == -1])['score']
    for ci in range(-1, len(vs)):
        rr = [r for r in allrows if r['cfg'] == ci]; s = summarise(rr)
        print(ci, vs[ci] if ci >= 0 else 'base', 'score %.6f d=%+.6f edgeJ %.5f div %d/%d/%d vetoed %d' % (s['score'], s['score'] - base, s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r.get('vetoed', 0) for r in rr)), flush=True)
