"""Independent rescoring of submission CSVs with the official metric (rescore reviewer).
Faithful path: CSV -> build_graph_from_rows (official) -> save_graph geff -> IndexedRXGraph.from_geff
-> tracking_cellmot.metrics.evaluate / node_recall / per_sample_metrics / summarise.
usage: python rescore.py cfg:set [cfg:set ...]
"""
import os, sys, json, time, warnings
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/workspace/official/src")
sys.path.insert(0, "/workspace/official/scripts")
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import polars as pl

OUT = Path("/workspace/verify/rescore")
GT = Path("/workspace/data/train")
DEFAULT = (1.625, 0.40625, 0.40625)
LISTS = {s: f"/workspace/{s}.txt" for s in ["hold36", "preview4", "audit32", "t127a", "t127b"]}
LISTS["prev4"] = "/workspace/preview4.txt"


def sanity(name, nodes, edges, T):
    issues = []
    nid = nodes["node_id"].to_numpy()
    if len(np.unique(nid)) != len(nid):
        issues.append("dup node_id")
    for c in ["t", "z", "y", "x"]:
        if nodes[c].null_count() or (nodes[c].cast(pl.Float64).is_nan().sum()):
            issues.append(f"null/nan {c}")
    tt = nodes["t"].to_numpy()
    present = set(np.unique(tt).tolist())
    missing = [t for t in range(T) if t not in present]
    if missing:
        issues.append(f"frames without nodes: {missing[:10]} (n={len(missing)})")
    if tt.min() < 0 or tt.max() > T - 1:
        issues.append(f"t out of range {tt.min()}..{tt.max()} T={T}")
    if (nodes["source_id"] != -1).any() or (nodes["target_id"] != -1).any():
        issues.append("node row with src/tgt != -1")
    if edges.height:
        s = edges["source_id"].to_numpy(); d = edges["target_id"].to_numpy()
        tmap = dict(zip(nid.tolist(), tt.tolist()))
        miss = [x for x in np.concatenate([s, d]).tolist() if x not in tmap]
        if miss:
            issues.append(f"edge endpoint not a node: {len(miss)}")
        else:
            dt = np.array([tmap[b] - tmap[a] for a, b in zip(s.tolist(), d.tolist())])
            if (dt != 1).any():
                issues.append(f"non-consecutive edges: {(dt != 1).sum()}")
        if (s == d).any():
            issues.append("self loop")
        pairs = np.stack([s, d], 1)
        if len(np.unique(pairs, axis=0)) != len(pairs):
            issues.append(f"dup edges: {len(pairs) - len(np.unique(pairs, axis=0))}")
        _, ind = np.unique(d, return_counts=True)
        if ind.max() > 1:
            issues.append(f"in-degree>1: {(ind > 1).sum()}")
        _, outd = np.unique(s, return_counts=True)
        if outd.max() > 2:
            issues.append(f"out-degree>2: {(outd > 2).sum()}")
        n_div = int((outd == 2).sum())
    else:
        n_div = 0
    return issues, n_div


def job(args):
    tag, name, nodes, edges = args
    warnings.filterwarnings("ignore")
    import tracksdata as td
    from geff import GeffMetadata
    from tracking_cellmot.io import open_dataset, save_graph
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics
    from csv_to_geffs import build_graph_from_rows
    t0 = time.time()
    ds = open_dataset(GT / name, load_image=False)
    scale = tuple(float(v) for v in ds.scale)
    T = ds.image_shape[0]
    issues, n_div_pred = sanity(name, nodes, edges, T)
    g = build_graph_from_rows(nodes, edges)
    gp = OUT / "geffs" / tag / f"{name}.geff"
    gp.parent.mkdir(parents=True, exist_ok=True)
    save_graph(g, gp, overwrite=True)
    pred = td.graph.IndexedRXGraph.from_geff(gp)
    pred = pred[0] if isinstance(pred, tuple) else pred
    gt = td.graph.IndexedRXGraph.from_geff(GT / f"{name}.geff")
    gt = gt[0] if isinstance(gt, tuple) else gt
    er = evaluate(pred, gt, scale=scale, max_distance=7.0)
    rec = node_recall(pred, gt) if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0
    meta = GeffMetadata.read(GT / f"{name}.geff")
    n_total = float((meta.extra or {}).get("estimated_number_of_nodes", float("nan")))
    row = per_sample_metrics(er, n_total, rec)
    row.update(movie=name, n_total=n_total, scale=list(scale), scale_is_default=bool(np.allclose(scale, DEFAULT)),
               T=T, issues=issues, n_pred_nodes_csv=nodes.height, n_pred_edges_csv=edges.height,
               n_pred_div_csv=n_div_pred, gt_nodes=gt.num_nodes(), gt_edges=gt.num_edges(), sec=time.time() - t0)
    return tag, row


def main():
    from tracking_cellmot.metrics import summarise
    specs = sys.argv[1:]
    jobs, meta = [], {}
    for spec in specs:
        cfg, st = spec.split(":")
        tag = f"{cfg}_{st}"
        csv = f"/workspace/cl/ps_{cfg}_{st}/submission.csv"
        df = pl.read_csv(csv)
        glob_issues = []
        if df.columns != ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]:
            glob_issues.append(f"columns {df.columns}")
        if not (df["id"].to_numpy() == np.arange(df.height)).all():
            glob_issues.append("ids not consecutive 0..N-1")
        rt = set(df["row_type"].unique().to_list())
        if rt - {"node", "edge"}:
            glob_issues.append(f"row_types {rt}")
        want = [l.strip() for l in open(LISTS[st]) if l.strip()]
        have = df["dataset"].unique().to_list()
        if set(want) != set(have):
            glob_issues.append(f"movie set mismatch missing={sorted(set(want)-set(have))} extra={sorted(set(have)-set(want))}")
        meta[tag] = dict(csv=csv, rows=df.height, n_movies=len(have), n_list=len(want), issues=glob_issues)
        for (name,), g in df.group_by("dataset"):
            n = g.filter(pl.col("row_type") == "node").sort("id")
            e = g.filter(pl.col("row_type") == "edge").sort("id")
            jobs.append((tag, name, n, e))
    jobs.sort(key=lambda j: -j[2].height)
    res = {}
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(22, len(jobs))) as pool:
        for tag, row in pool.imap_unordered(job, jobs):
            res.setdefault(tag, []).append(row)
            print("done", tag, row["movie"], round(row["sec"], 1), flush=True)
    for tag in res:
        rows = sorted(res[tag], key=lambda r: r["movie"])
        s = summarise(rows)
        meta[tag]["summary"] = s
        json.dump(dict(meta=meta[tag], rows=rows), open(OUT / f"res_{tag}.json", "w"), indent=1, default=str)
        bad = [(r["movie"], r["issues"]) for r in rows if r["issues"]]
        print(f"{tag}: n={s['n']} score={s['score']:.6f} adjE={s['adj_edge_jaccard']:.6f} E={s['edge_jaccard']:.6f} "
              f"divJ={s['division_jaccard']:.6f} div={s['division_tp']}/{s['division_fp']}/{s['division_fn']} "
              f"recall={s['node_recall']:.4f} global_issues={meta[tag]['issues']} movie_issues={bad} "
              f"nondefault_scale={[r['movie'] for r in rows if not r['scale_is_default']]}", flush=True)


if __name__ == "__main__":
    main()
