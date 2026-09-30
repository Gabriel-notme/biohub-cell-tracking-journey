import os, sys, json, glob, time
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
gdir, cdir, ddir, outp = sys.argv[1:5]; grid = json.loads(open(sys.argv[5]).read()); nproc = int(sys.argv[6])

def dn_apply(nodes, edges, cands, dn, ts=0.9, tst=0.97, dth=1.1, ts_low=1.1, tst_low=1.1, veto_d=-1, veto_fork=1.1, prune_min_len=2, top1=False):
    from div_complete import structure
    def ok(c):
        d = dn.get(c['p'], 0.0); f = c['fork']
        if c['typ'] == 'start': acc = f >= ts or (d >= dth and f >= ts_low)
        else: acc = f >= tst or (d >= dth and f >= tst_low)
        if acc and d < veto_d and f < veto_fork: acc = False
        return acc
    cs = [c for c in cands if ok(c)]
    if top1:
        best = {}
        for c in cs:
            if c['p'] not in best or c['fork'] > best[c['p']]['fork']: best[c['p']] = c
        cs = list(best.values())
    cs.sort(key=lambda c: -(c['fork'] + dn.get(c['p'], 0.0)))
    used_p = set(); used_b = set(); rm = set(); add = []
    for c in cs:
        p, a, b, q = c['p'], c['a'], c['b'], c['q']
        if p in used_p or b in used_b or a in used_b: continue
        if c['typ'] == 'stolen':
            if q in used_p: continue
            rm.add((q, b)); used_p.add(q)
        used_p.add(p); used_b.add(b); used_b.add(a); add.append((p, b))
    ne = [e for e in edges if (int(e['source_id']), int(e['target_id'])) not in rm] + [{'source_id': p, 'target_id': b, 'div_complete': 1} for p, b in add]
    st = {'div_added': len(add), 'div_stolen': len(rm)}
    nn = nodes
    if prune_min_len:
        import prune
        nn, ne, pst = prune.prune_fragments(nodes, ne, prune_min_len); st.update(pst)
    return nn, ne, st

def job(name):
    import evalx
    nodes, edges = evalx.load_graph_json(gdir + '/' + name + '.json')
    cands = json.load(open(cdir + '/' + name + '.cands.json'))
    dn = {int(k): v for k, v in json.load(open(ddir + '/' + name + '.divnet.json')).items()}
    rows = []
    r = evalx.score_movie(name, nodes, edges); r['cfg'] = -1; rows.append(r)
    for ci, cfg in enumerate(grid):
        nn, ne, st = dn_apply(nodes, edges, cands, dn, **cfg)
        r = evalx.score_movie(name, nn, ne); r['cfg'] = ci; r['mod_stats'] = st; rows.append(r)
    return rows

if __name__ == '__main__':
    names = sorted(os.path.basename(p).split('.')[0] for p in glob.glob(ddir + '/*.divnet.json'))
    with Pool(nproc) as pool: allrows = [r for rs in pool.map(job, names, chunksize=1) for r in rs]
    from tracking_cellmot.metrics import summarise
    base = summarise([r for r in allrows if r['cfg'] == -1])['score']
    out = {}
    for ci in range(-1, len(grid)):
        rr = [r for r in allrows if r['cfg'] == ci]; s = summarise(rr); tot = {}
        for r in rr:
            for k, v in (r.get('mod_stats') or {}).items(): tot[k] = tot.get(k, 0) + v
        out[ci] = {'summary': s, 'cfg': grid[ci] if ci >= 0 else 'baseline', 'stats': tot}
        print(ci, 'score %.6f d=%+.6f edgeJ %.5f adj %.5f div %d/%d/%d' % (s['score'], s['score'] - base, s['edge_jaccard'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn']), json.dumps(out[ci]['cfg']), tot, flush=True)
    json.dump(out, open(outp, 'w'))
    print('movies', len(names))
