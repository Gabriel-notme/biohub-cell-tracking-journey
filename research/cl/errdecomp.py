"""Error decomposition of edge FP/FN for given movies (rescore reviewer, read-only on inputs)."""
import os, sys, json, warnings
os.environ["POLARS_MAX_THREADS"] = "2"
sys.path.insert(0, "/workspace/official/src")
warnings.filterwarnings("ignore")
import numpy as np, polars as pl, tracksdata as td
from collections import Counter
from tracking_cellmot.metrics import _evaluate, _evaluate_matched_graph
from tracking_cellmot.io import open_dataset

K = td.DEFAULT_ATTR_KEYS
tag = sys.argv[1]
for name in sys.argv[2:]:
    pred = td.graph.IndexedRXGraph.from_geff(f"/workspace/verify/rescore/geffs/{tag}/{name}.geff")
    pred = pred[0] if isinstance(pred, tuple) else pred
    gt = td.graph.IndexedRXGraph.from_geff(f"/workspace/data/train/{name}.geff")
    gt = gt[0] if isinstance(gt, tuple) else gt
    scale = open_dataset(f"/workspace/data/train/{name}", load_image=False).scale
    _evaluate(pred, gt, "jaccard", scale, 7.0)
    ea = _evaluate_matched_graph(pred, gt)
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID, K.T, "z", "y", "x"])
    p2g = {a: b for a, b in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list()) if b is not None and b != -1}
    g2p = {}
    for a, b in p2g.items():
        g2p.setdefault(b, []).append(a)
    ptime = dict(zip(na[K.NODE_ID].to_list(), na[K.T].to_list()))
    gna = gt.node_attrs(attr_keys=[K.NODE_ID, K.T])
    gtime = dict(zip(gna[K.NODE_ID].to_list(), gna[K.T].to_list()))
    gea = gt.edge_attrs(attr_keys=[])
    gedges = list(zip(gea[K.EDGE_SOURCE].to_list(), gea[K.EDGE_TARGET].to_list()))
    gout = Counter(s for s, _ in gedges)
    pea = pred.edge_attrs(attr_keys=[])
    pset = set(zip(pea[K.EDGE_SOURCE].to_list(), pea[K.EDGE_TARGET].to_list()))
    pout = {}
    pin = {}
    for s, t in pset:
        pout.setdefault(s, []).append(t); pin[t] = s
    matched_pairs = set()
    tp_rows = ea.filter(pl.col(K.MATCHED_EDGE_MASK))
    for s, t in zip(tp_rows[K.EDGE_SOURCE].to_list(), tp_rows[K.EDGE_TARGET].to_list()):
        matched_pairs.add((p2g[s], p2g[t]))
    fn = Counter(); fn_t = []
    for u, v in gedges:
        if (u, v) in matched_pairs:
            continue
        fn_t.append(gtime[u])
        if u not in g2p and v not in g2p:
            fn["both GT nodes undetected"] += 1
        elif u not in g2p:
            fn["source GT node undetected"] += 1
        elif v not in g2p:
            fn["target GT node undetected"] += 1
        else:
            pu = g2p[u]; pv = g2p[v]
            if len(pu) > 1 or len(pv) > 1:
                fn["multi-match (merge)"] += 1
            kids = [c for p in pu for c in pout.get(p, [])]
            par = [pin.get(p) for p in pv]
            if not kids and all(x is None for x in par):
                fn["link missing: pred track ends at u and starts at v"] += 1
            elif not kids:
                fn["link missing: pred u has no child, v has other parent"] += 1
            elif all(x is None for x in par):
                fn["link missing: v has no parent, u linked elsewhere"] += 1
            else:
                fn["switch: u linked elsewhere AND v has other parent"] += 1
    fp = Counter(); fp_t = []
    fps = ea.filter(pl.col("pred_valid") & ~pl.col(K.MATCHED_EDGE_MASK))
    for s, t in zip(fps[K.EDGE_SOURCE].to_list(), fps[K.EDGE_TARGET].to_list()):
        fp_t.append(ptime[s])
        gs, gtn = p2g.get(s), p2g.get(t)
        isdiv = len(pout.get(s, [])) == 2
        pre = "[pred-division] " if isdiv else ""
        if gs is None:
            fp[pre + "source unmatched, target matched GT node w/ parent"] += 1
        elif gtn is None:
            fp[pre + "target unmatched (pred went to non-GT node)"] += 1
        else:
            fp[pre + "both matched but to non-linked GT nodes (ID switch)"] += 1
    tp = int(ea[K.MATCHED_EDGE_MASK].sum())
    print(f"=== {tag} {name}: gtE={len(gedges)} TP={tp} FP={len(fp_t)} FN={len(fn_t)} gt_div={sum(1 for c in gout.values() if c==2)}")
    for k, v in fn.most_common(): print(f"  FN {v:4d} {k}")
    for k, v in fp.most_common(): print(f"  FP {v:4d} {k}")
    gt_t = np.array([gtime[u] for u, _ in gedges])
    bins = [0, 25, 50, 75, 100]
    print("  GT edges by t-bin", np.histogram(gt_t, bins)[0].tolist(), " FN by t-bin", np.histogram(fn_t, bins)[0].tolist(), " FP by t-bin", np.histogram(fp_t, bins)[0].tolist())
    pn = np.bincount(na[K.T].to_numpy(), minlength=100)
    print("  pred nodes/frame t=0,25,50,75,99:", [int(pn[i]) for i in (0, 25, 50, 75, 99)])
