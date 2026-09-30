"""lin_run.py <ref_graph_dir> <out_dir> <data_dir> <nproc> <gpus comma>: B5 lineage stages (LineagePipeline.refine, exactly as
lin_ablate.py drop=none, which reproduced B5's lineage graphs) on every reference graph in ref_graph_dir."""
import os, sys, json, time
from pathlib import Path
from multiprocessing import get_context
ART = '/workspace/art_b56/artifact_bundle'
_G = {}
REF, OUT, DATA = sys.argv[1], sys.argv[2], sys.argv[3]


def init(gpus):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[(mp.current_process()._identity[0] - 1) % len(gpus)])
    for k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']: os.environ[k] = '2'
    os.environ['BIOHUB_ART'] = ART; os.environ['BIOHUB_BASE_REPO'] = '/workspace/models/support/repo'


def job(name):
    dst = Path(OUT) / (name + '.json')
    if dst.exists(): return name, 'cached', 0.0
    if 'p' not in _G:
        sys.path.insert(0, ART)
        from lineage_pipeline import LineagePipeline
        sel = json.load(open(ART + '/final_selection.json'))['B5']
        _G['p'] = LineagePipeline(ART, sel, DATA, '/workspace/cache/linrun_%d' % os.getpid())
    t0 = time.time()
    d = json.load(open(Path(REF) / (name + '.json')))
    nodes = {int(k): v for k, v in d['nodes'].items()}
    try:
        nodes, edges, stats = _G['p'].refine(name, nodes, d['edges'])
    except Exception as e:
        return name, 'error ' + repr(e)[:200], time.time() - t0
    tmp = dst.with_suffix('.tmp'); tmp.write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges, 'lineage': stats})); os.replace(tmp, dst)
    return name, 'ok', time.time() - t0


if __name__ == '__main__':
    nproc = int(sys.argv[4]); gpus = [int(x) for x in sys.argv[5].split(',')]
    Path(OUT).mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in Path(REF).glob('*.json'))
    names.sort(key=lambda n: -(Path(REF) / (n + '.json')).stat().st_size)
    t0 = time.time(); n = 0
    with get_context('spawn').Pool(nproc, initializer=init, initargs=(gpus,)) as pool:
        for name, st, sec in pool.imap_unordered(job, names):
            n += 1; print('LIN', n, len(names), name, st, round(sec, 1), round(time.time() - t0), flush=True)
    print('LIN_DONE', len(names), round(time.time() - t0))
