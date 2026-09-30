"""Fast evaluation of learned division completion using cached candidate rows.
B5 base -> learned dc (th) -> transplant P3's dsr additions -> dfork K2 -> prune 2 [-> relink th2] [-> edge_link th3]
usage: div_eval.py <cfg> <model: full|nob1> <th> [relink_th or -] [el_th or -]"""
import os, sys, json
os.environ.setdefault('POLARS_MAX_THREADS', '2'); os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, '/workspace/official/src'); sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/p12ds'); sys.path.insert(0, '/workspace/cl')
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
cfg, model, th = sys.argv[1], sys.argv[2], float(sys.argv[3])
rl_th = sys.argv[4] if len(sys.argv) > 4 else '-'
el_th = sys.argv[5] if len(sys.argv) > 5 else '-'
SETS = {'hold36': ('/workspace/hold36.txt', '/workspace/runs/b5f_hold36/working/lineage_graphs', '/workspace/runs/p3_hold36/graphs', '/workspace/runs/fullgraph_hold36'),
        'prev4': ('/workspace/preview4.txt', '/workspace/runs/b5f_prev4/working/lineage_graphs', '/workspace/runs/p3_prev4/graphs', '/workspace/runs/fullgraph_prev4'),
        'audit32': ('/workspace/audit32.txt', '/workspace/sync3/runs/b5f_audit32/working/lineage_graphs', '/workspace/sync3/runs/p3_audit32/graphs', '/workspace/sync3/runs/fullgraph_audit32')}
ONLY = os.environ.get('ONLY_SETS')


def job(a):
    s, name, bdir, pdir, fdir = a
    import evalx, div_learn, dfork, prune, edge_link, relink
    nodes, edges = evalx.load_graph_json(Path(bdir) / (name + '.json'))
    cands = json.load(open('/workspace/cl/cands/%s__%s.json' % (s, name)))
    byp = defaultdict(list); byb = defaultdict(list)
    for c in cands: byp[c['p']].append(c['fork']); byb[c['b']].append(c['fork'])
    for c in cands:
        sp = sorted(byp[c['p']], reverse=True); sb = sorted(byb[c['b']], reverse=True)
        c['fork_rank_p'] = sp.index(c['fork']); c['fork_rank_b'] = sb.index(c['fork'])
        c['fork_gap_p'] = c['fork'] - (sp[1] if len(sp) > 1 and sp[0] == c['fork'] else sp[0]); c['is_stolen'] = int(c['typ'] == 'stolen')
    keep = [c for c in cands if c['fork'] >= 0.05]
    prob = div_learn.predict(keep, '/workspace/cl/div_lgb_%s.txt' % model)
    edges, st = div_learn.apply(nodes, edges, keep, prob, th=th)
    # transplant P3 dsr additions
    pn, pe = evalx.load_graph_json(Path(pdir) / (name + '.json'))
    succ = defaultdict(list); par = {}
    for e in edges: succ[int(e['source_id'])].append(int(e['target_id'])); par[int(e['target_id'])] = int(e['source_id'])
    dsr_e = [e for e in pe if e.get('dsr')]
    newn = {int(e['target_id']) for e in dsr_e if int(e['target_id']) not in nodes}
    nodes = dict(nodes); nd = 0
    for e in dsr_e:
        s_, t_ = int(e['source_id']), int(e['target_id'])
        if t_ in newn and s_ in nodes and len(succ.get(s_, [])) == 1 and t_ not in nodes:
            nodes[t_] = pn[t_]; edges.append({'source_id': s_, 'target_id': t_, 'dsr': 1}); succ[s_].append(t_); par[t_] = s_; nd += 1
    for e in dsr_e:
        s_, t_ = int(e['source_id']), int(e['target_id'])
        if s_ in newn and s_ in nodes and t_ in nodes and t_ not in par and not succ.get(s_):
            edges.append({'source_id': s_, 'target_id': t_, 'dsr': 1}); succ[s_].append(t_); par[t_] = s_
    st['dsr_transplanted'] = nd
    edges, x = dfork.resolve(nodes, edges, K=2); st.update(x)
    nodes, edges, x = prune.prune_fragments(nodes, edges, 2); st.update({k: v for k, v in x.items() if isinstance(v, (int, float))})
    fp = Path(fdir) / (name + '.geff')
    if rl_th != '-':
        nodes, edges, x = relink.apply(nodes, edges, edge_link.load_full(fp), '/workspace/cl/relink_lgb.txt', th=float(rl_th)); st.update(x)
    if el_th != '-':
        nodes, edges, x = edge_link.apply(nodes, edges, fp, '/workspace/cl/edge_lgb.txt', th=float(el_th)); st.update(x)
    pairs = [(int(e['source_id']), int(e['target_id'])) for e in edges]
    assert len(pairs) == len(set(pairs)) and max(Counter(t for _, t in pairs).values()) <= 1 and max(Counter(s_ for s_, _ in pairs).values()) <= 2
    assert all(int(nodes[t]['t']) == int(nodes[s_]['t']) + 1 for s_, t in pairs)
    od = Path('/workspace/cl/out_%s/%s' % (cfg, s)); od.mkdir(parents=True, exist_ok=True)
    (od / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges}))
    r = evalx.score_movie(name, nodes, edges); r['movie'] = name; r['stats'] = st
    return s, r


if __name__ == '__main__':
    jobs = []
    for s, (lst, b, p, f) in SETS.items():
        if ONLY and s not in ONLY.split(','): continue
        for n in [l.strip() for l in open(lst) if l.strip()]: jobs.append((s, n, b, p, f))
    with Pool(36) as pool: res = pool.map(job, jobs)
    tot = Counter()
    for s in SETS:
        rows = [r for ss, r in res if ss == s]
        if not rows: continue
        for r in rows: tot.update({k: v for k, v in r['stats'].items() if isinstance(v, (int, float))})
        json.dump(rows, open('/workspace/cl/rows/%s_%s.json' % (cfg, s), 'w'))
    print(cfg, model, th, rl_th, el_th, dict(tot))
