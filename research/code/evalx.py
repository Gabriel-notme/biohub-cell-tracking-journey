import os, sys, json
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/official/src')
os.environ.setdefault('POLARS_MAX_THREADS', '4')
import numpy as np
import polars as pl
import tracksdata as td
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
TRAIN = Path('/workspace/data/train')
SCALE = (1.625, 0.40625, 0.40625)
K = td.DEFAULT_ATTR_KEYS

def load_graph_json(path):
    d = json.loads(Path(path).read_text())
    return {int(k): v for k, v in d['nodes'].items()}, d['edges']

def to_graph(nodes, edges, rounding=True):
    g = td.graph.InMemoryGraph()
    for k in ['z', 'y', 'x']: g.add_node_attr_key(k, pl.Float64, 0.)
    old = list(nodes)
    if rounding:
        rows = [{'t': int(nodes[n]['t']), **{k: float(max(0, int(round(nodes[n][k])))) for k in ['z', 'y', 'x']}} for n in old]
    else:
        rows = [{'t': int(nodes[n]['t']), **{k: float(max(0., nodes[n][k])) for k in ['z', 'y', 'x']}} for n in old]
    assigned = g.bulk_add_nodes(rows)
    mapping = dict(zip(old, assigned))
    if edges: g.bulk_add_edges([{'source_id': mapping[int(e['source_id'])], 'target_id': mapping[int(e['target_id'])]} for e in edges])
    return g, mapping

def load_gt(name):
    gtpath = TRAIN / (name + '.geff'); gt = td.graph.IndexedRXGraph.from_geff(gtpath)
    if isinstance(gt, tuple): gt = gt[0]
    n_total = float((GeffMetadata.read(gtpath).extra or {}).get('estimated_number_of_nodes', float('nan')))
    return gt, n_total

def score_movie(name, nodes, edges, rounding=True, detail=False):
    gt, n_total = load_gt(name)
    pred, mapping = to_graph(nodes, edges, rounding)
    er = evaluate(pred, gt, scale=SCALE, max_distance=7.)
    row = per_sample_metrics(er, n_total, node_recall(pred, gt)); row['movie'] = name; row['n_total'] = n_total
    if detail:
        row['analysis'] = analyze(pred, gt, mapping, nodes, edges)
    return row

def analyze(pred, gt, mapping, nodes, edges):
    inv = {v: k for k, v in mapping.items()}
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    p2g = {inv[int(a)]: int(b) for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and int(b) != -1}
    g2p = {g: p for p, g in p2g.items()}
    gids = gt.node_ids(); ga = gt.node_attrs(attr_keys=[K.NODE_ID, 't'])
    gt_edges = [(int(s), int(d)) for s, d in zip(*[gt.edge_attrs()[c].to_list() for c in [K.EDGE_SOURCE, K.EDGE_TARGET]])]
    succ = defaultdict(list); pred_par = {}
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id']); succ[s].append(d); pred_par[d] = s
    gsucc = defaultdict(list)
    for s, d in gt_edges: gsucc[s].append(d)
    cat = Counter()
    for s, d in gt_edges:
        ps, pd_ = g2p.get(s), g2p.get(d)
        if ps is None and pd_ is None: cat['fn_both_missing'] += 1; continue
        if ps is None: cat['fn_src_missing'] += 1; continue
        if pd_ is None: cat['fn_dst_missing'] += 1; continue
        if pd_ in succ.get(ps, []): cat['tp'] += 1; continue
        if len(succ.get(ps, [])) == 0 and pd_ not in pred_par: cat['fn_gap_both_free'] += 1
        elif len(succ.get(ps, [])) == 0: cat['fn_src_ends_dst_taken'] += 1
        elif pd_ not in pred_par: cat['fn_src_elsewhere_dst_starts'] += 1
        else: cat['fn_swap'] += 1
    # FP analysis: predicted edges from matched source with GT successors
    for e in edges:
        s, d = int(e['source_id']), int(e['target_id'])
        gs = p2g.get(s)
        if gs is not None and gsucc.get(gs):
            gd = p2g.get(d)
            if gd is not None and gd in gsucc[gs]: continue
            if gd is None: cat['fp_to_unmatched'] += 1
            else: cat['fp_to_other_gt'] += 1
    return dict(cat)
