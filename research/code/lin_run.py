"""Run B5 lineage pipeline (+ optional P1 stage) on transformed input graphs and score.
usage: lin_run.py <variant_name> <movies.txt> <out_dir> <nproc> [transform] [pconfig]
transform: none | reinsert:<minlen>   (re-insert raw components dropped by base post-processing)
"""
import os, sys, json, time, glob
from pathlib import Path
from collections import defaultdict
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'
P12 = '/workspace/p12ds'
RUNW = os.environ.get('LIN_RUNW', '/workspace/runs/b5_hold36/working')
DATA = os.environ.get('LIN_DATA', '/workspace/data/train')
variant, movies_txt, out_dir, nproc = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
transform = sys.argv[5] if len(sys.argv) > 5 else 'none'
pconfig = sys.argv[6] if len(sys.argv) > 6 else P12 + '/p1_config.json'
_G = {}
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
    os.environ['POLARS_MAX_THREADS'] = '2'
    os.environ['BIOHUB_ART'] = ART
    os.environ.setdefault('BIOHUB_BASE_REPO', RUNW + '/tracking_repo')
def pipe():
    if 'p' not in _G:
        sys.path.insert(0, ART); sys.path.insert(0, P12)
        from lineage_pipeline import LineagePipeline
        from refine_events import EventRefiner
        sel = json.load(open(ART + '/final_selection.json'))[os.environ.get('LIN_SEL', 'B5')]
        _G['p'] = LineagePipeline(ART, sel, DATA, '/workspace/cache/lin_' + variant)
        cfg = json.load(open(pconfig)); _G['cfg'] = cfg
        _G['er'] = EventRefiner([Path(ART) / n for n in cfg['models']], DATA, cfg.get('event_config'), Path('/workspace/cache/linp_' + variant))
    return _G['p'], _G['er'], _G['cfg']
def load_input(name):
    d = json.load(open(RUNW + '/reference_graphs/%s.json' % name))
    nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
    info = {}
    if transform.startswith('reinsert'):
        import zarr, numpy as np
        minlen = int(transform.split(':')[1])
        g = zarr.open_group(RUNW + '/tracking_repo/predictions/unknown/unet_transformer/split_0/%s.geff' % name, mode='r')
        ids = np.asarray(g['nodes/ids'][:]); P = {k: np.asarray(g['nodes/props/%s/values' % k][:]) for k in 'tzyx'}
        E = np.asarray(g['edges/ids'][:]); ep = np.asarray(g['edges/props/edge_prob/values'][:])
        raw = {int(i): j for j, i in enumerate(ids)}
        missing = set(raw) - set(nodes)
        # components among missing nodes using raw edges
        adj = defaultdict(list); redges = []
        for (s, t), p in zip(E.tolist(), ep.tolist()):
            if s in missing and t in missing: adj[s].append(t); adj[t].append(s); redges.append((s, t, p))
        seen = set(); add = set()
        for n in missing:
            if n in seen: continue
            comp = []; st = [n]; seen.add(n)
            while st:
                x = st.pop(); comp.append(x)
                for y in adj[x]:
                    if y not in seen: seen.add(y); st.append(y)
            if len(comp) >= minlen: add.update(comp)
        import numpy as np
        S = np.array([1.625, .40625, .40625])
        for n in add:
            j = raw[n]; nodes[n] = {'node_id': n, 't': int(P['t'][j]), 'z': float(P['z'][j]), 'y': float(P['y'][j]), 'x': float(P['x'][j])}
        for s, t, p in redges:
            if s in add and t in add:
                a, b = nodes[s], nodes[t]
                dist = float(np.linalg.norm((np.array([a['z'], a['y'], a['x']]) - np.array([b['z'], b['y'], b['x']])) * S))
                edges.append({'source_id': s, 'target_id': t, 'edge_prob': float(p), 'distance_um': dist})
        info = {'reinserted_nodes': len(add), 'missing': len(missing)}
    return nodes, edges, info
def job(name):
    import evalx
    p, er, cfg = pipe()
    import div_complete as dc, prune
    t0 = time.time()
    nodes, edges, info = load_input(name)
    rows = []
    try:
        nodes, edges, stats = p.refine(name, nodes, edges)
        r = evalx.score_movie(name, nodes, edges); r.update(cfg_name='lin', movie=name, info=info, sec=time.time() - t0); rows.append(r)
        (out_dir / 'lin').mkdir(parents=True, exist_ok=True)
        (out_dir / 'lin' / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges}))
        cands = dc.score_candidates(er, name, nodes, edges)
        ne, st = dc.apply(nodes, edges, cands, **cfg['apply'])
        n2, ne, pst = prune.prune_fragments(nodes, ne, int(cfg.get('prune_min_len', 0) or 0)) if cfg.get('prune_min_len') else (nodes, ne, {})
        r = evalx.score_movie(name, n2, ne); r.update(cfg_name='p', movie=name, info=dict(info, **st), sec=time.time() - t0); rows.append(r)
        (out_dir / 'p').mkdir(parents=True, exist_ok=True)
        (out_dir / 'p' / (name + '.json')).write_text(json.dumps({'nodes': {str(k): v for k, v in n2.items()}, 'edges': ne}))
    except Exception as e:
        import traceback; traceback.print_exc(); rows.append({'movie': name, 'error': repr(e)})
    print('DONE', name, round(time.time() - t0, 1), flush=True)
    return rows
if __name__ == '__main__':
    names = [l.strip() for l in open(movies_txt) if l.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    with get_context('spawn').Pool(nproc, initializer=init, initargs=([0, 1],)) as pool:
        rows = [r for rs in pool.imap_unordered(job, names) for r in rs]
    json.dump(rows, open(out_dir / 'rows.json', 'w'))
    from tracking_cellmot.metrics import summarise
    for c in ['lin', 'p']:
        rr = [r for r in rows if r.get('cfg_name') == c]
        if not rr: continue
        s = summarise(rr)
        print(variant, c, 'n=%d score %.6f adjE %.6f E %.6f div %d/%d/%d nodes %d' % (len(rr), s['score'], s['adj_edge_jaccard'], s['edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r['num_pred_nodes'] for r in rr)), flush=True)
    for r in rows:
        if 'error' in r: print('ERR', r['movie'], r['error'])
