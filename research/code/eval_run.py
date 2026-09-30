import os, sys, json, glob
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '/workspace/code')
run = Path(sys.argv[1]); nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
detail_stages = set(sys.argv[3].split(',')) if len(sys.argv) > 3 else set()
out = run / 'eval'; out.mkdir(exist_ok=True)

def job(args):
    stage, path = args
    import evalx
    name = Path(path).stem
    dest = out / stage / (name + '.json')
    if dest.exists(): return json.loads(dest.read_text())
    nodes, edges = evalx.load_graph_json(path)
    row = evalx.score_movie(name, nodes, edges, detail=(stage in detail_stages or 'all' in detail_stages))
    row['stage'] = stage
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(row))
    return row

jobs = []
w = run / 'working'
for p in sorted((w / 'reference_graphs').glob('*.json')): jobs.append(('ref', str(p)))
for sd in sorted((run / 'stages').glob('*')):
    for p in sorted(sd.glob('*.json')): jobs.append((sd.name, str(p)))
for p in sorted((w / 'lineage_graphs').glob('*.json')): jobs.append(('final', str(p)))
with Pool(nproc) as pool: rows = pool.map(job, jobs, chunksize=1)
sys.path.insert(0, '/workspace/official/src')
from tracking_cellmot.metrics import summarise
by = {}
for r in rows: by.setdefault(r['stage'], []).append(r)
summ = {s: dict(summarise(v), movies=len(v)) for s, v in by.items()}
json.dump({'summary': summ}, open(out / 'summary.json', 'w'), indent=1)
for s in sorted(summ): print(s, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in summ[s].items()}))
