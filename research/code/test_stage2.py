import os, sys, json, glob, time
from pathlib import Path
from multiprocessing import get_context
sys.path.insert(0, '/workspace/code'); sys.path.insert(0, '/workspace/official/src')
ART = '/workspace/art_b56/artifact_bundle'
gdir, outp, kinds = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 3
def init(gl):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gl[(mp.current_process()._identity[0] - 1) % len(gl)])
def job(path):
    sys.path.insert(0, ART)
    import evalx
    from lineage_pipeline import refiner_class
    sel = json.load(open(ART + '/final_selection.json'))['B5']
    stages = {s['kind']: s for s in sel['stages']}
    name = Path(path).stem
    nodes, edges = evalx.load_graph_json(path)
    rows = [dict(evalx.score_movie(name, nodes, edges), cfg='base')]
    for kind in kinds:
        st = stages[kind]
        obj = refiner_class(kind)([Path(ART) / n for n in st['model_files']], '/workspace/data/train', st['config'], Path('/workspace/cache/stage2_' + kind))
        t0 = time.time()
        nn, ee, rep = obj.refine(name, {k: dict(v) for k, v in nodes.items()}, [dict(e) for e in edges])
        r = evalx.score_movie(name, nn, ee); r['cfg'] = kind; r['sec'] = time.time() - t0; rows.append(r)
    return rows
if __name__ == '__main__':
    paths = sorted(glob.glob(gdir + '/*.json'))
    with get_context('spawn').Pool(nproc, initializer=init, initargs=([0, 1],)) as pool:
        allrows = [r for rs in pool.map(job, paths, chunksize=1) for r in rs]
    from tracking_cellmot.metrics import summarise
    base = summarise([r for r in allrows if r['cfg'] == 'base'])['score']
    for k in ['base'] + kinds:
        rr = [r for r in allrows if r['cfg'] == k]; s = summarise(rr)
        print(k, 'score %.6f d=%+.6f edgeJ %.5f adj %.5f div %d/%d/%d sec %.0f' % (s['score'], s['score'] - base, s['edge_jaccard'], s['adj_edge_jaccard'], s['division_tp'], s['division_fp'], s['division_fn'], sum(r.get('sec', 0) for r in rr)), flush=True)
    json.dump(allrows, open(outp, 'w'))
