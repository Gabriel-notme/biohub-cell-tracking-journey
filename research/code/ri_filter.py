import os, sys, json, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
os.environ['POLARS_MAX_THREADS'] = '2'
gdir = sys.argv[1]; LS = [int(x) for x in sys.argv[2].split(',')]; RUNW = sys.argv[3] if len(sys.argv) > 3 else '/workspace/runs/b5_hold36/working'
def comps(nodes, edges):
    adj = defaultdict(list); out = defaultdict(int)
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); adj[s].append(d); adj[d].append(s); out[s] += 1
    seen = {}; cs = []
    for n in nodes:
        if n in seen: continue
        st = [n]; seen[n] = len(cs); mem = [n]
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in seen: seen[y] = len(cs); st.append(y); mem.append(y)
        cs.append(mem)
    return cs, out
def job(a):
    p, L = a
    import evalx, zarr, numpy as np
    name = Path(p).stem
    nodes, edges = evalx.load_graph_json(p)
    ref = json.load(open(RUNW + '/reference_graphs/%s.json' % name)); refids = {int(k) for k in ref['nodes']}
    g = zarr.open_group(RUNW + '/tracking_repo/predictions/unknown/unet_transformer/split_0/%s.geff' % name, mode='r')
    rawids = set(int(i) for i in np.asarray(g['nodes/ids'][:]))
    rein = (rawids - refids) & set(nodes)
    drop = set()
    if L > 0:
        cs, out = comps(nodes, edges)
        for mem in cs:
            if all(m in rein for m in mem) and len(mem) < L and not any(out[m] > 1 for m in mem): drop.update(mem)
    nn = {k: v for k, v in nodes.items() if k not in drop}; ne = [e for e in edges if int(e['source_id']) not in drop and int(e['target_id']) not in drop]
    r = evalx.score_movie(name, nn, ne); r['L'] = L; r['rein'] = len(rein); r['dropped'] = len(drop)
    return r
if __name__ == '__main__':
    ps = sorted(glob.glob(gdir + '/*.json'))
    with Pool(32) as pool: rows = pool.map(job, [(p, L) for L in LS for p in ps], chunksize=1)
    from tracking_cellmot.metrics import summarise
    for L in LS:
        rr = [r for r in rows if r['L'] == L]; s = summarise(rr)
        print('L=%d score %.6f adjE %.6f E %.6f div %d/%d/%d nodes %d rein %d dropped %d' % (L, s['score'], s['adj_edge_jaccard'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r['num_pred_nodes'] for r in rr), sum(r['rein'] for r in rr), sum(r['dropped'] for r in rr)), flush=True)
