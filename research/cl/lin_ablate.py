"""Re-run B5's lineage stages from B5 reference graphs with one stage removed (ablation), writing lineage graphs that the P-stage
can consume. usage: lin_ablate.py <tag> <drop: none|stage kind> <sets comma> [nproc]
out: /workspace/cl/lin/<tag>/<set>/working/lineage_graphs/<movie>.json  (+ reference_graphs symlink next to it)"""
import os, sys, json, time
from pathlib import Path
from multiprocessing import get_context
ART = '/workspace/art_b56/artifact_bundle'
DATA = '/workspace/data/train'
RUNW = {'hold36': '/workspace/runs/b5f_hold36/working', 'prev4': '/workspace/runs/b5f_prev4/working', 'audit32': '/workspace/sync3/runs/b5f_audit32/working',
        't127a': '/workspace/sync4/runs/b5f_t127a/working', 't127b': '/workspace/sync3/runs/b5f_t127b/working'}
_G = {}


def init(gpus):
    import multiprocessing as mp
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[(mp.current_process()._identity[0] - 1) % len(gpus)])
    for k in ['POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']: os.environ[k] = '2'
    os.environ['BIOHUB_ART'] = ART


def job(args):
    tag, drop, s, name = args
    out = Path('/workspace/cl/lin/%s/%s/working/lineage_graphs' % (tag, s)); dst = out / (name + '.json')
    if dst.exists(): return name, 'cached', 0.0
    os.environ['BIOHUB_BASE_REPO'] = RUNW[s] + '/tracking_repo'
    if 'p' not in _G:
        sys.path.insert(0, ART)
        if os.environ.get('EVT_TTA', 'none') != 'none':
            sys.path.insert(0, '/workspace/cl'); import evt_tta; evt_tta.install()
        from lineage_pipeline import LineagePipeline
        sel = json.load(open(ART + '/final_selection.json'))['B5']
        if drop != 'none':
            sel = dict(sel); sel['stages'] = [st for st in sel['stages'] if st['kind'] != drop]
        _G['p'] = LineagePipeline(ART, sel, DATA, '/workspace/cache/abl_%s_%d' % (tag, os.getpid()))
    t0 = time.time()
    d = json.load(open(RUNW[s] + '/reference_graphs/%s.json' % name))
    nodes = {int(k): v for k, v in d['nodes'].items()}
    try:
        nodes, edges, stats = _G['p'].refine(name, nodes, d['edges'])
    except Exception as e:
        return name, 'error ' + repr(e)[:200], time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.tmp'); tmp.write_text(json.dumps({'nodes': {str(k): v for k, v in nodes.items()}, 'edges': edges, 'lineage': stats})); os.replace(tmp, dst)
    return name, 'ok', time.time() - t0


if __name__ == '__main__':
    tag, drop, sets = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    jobs = []
    for s in sets:
        base = Path('/workspace/cl/lin/%s/%s/working' % (tag, s)); base.mkdir(parents=True, exist_ok=True)
        ref = base / 'reference_graphs'
        if not ref.exists(): ref.symlink_to(RUNW[s] + '/reference_graphs')
        names = sorted(p.stem for p in Path(RUNW[s] + '/lineage_graphs').glob('*.json'))
        names.sort(key=lambda n: -Path(RUNW[s] + '/reference_graphs/%s.json' % n).stat().st_size)
        jobs += [(tag, drop, s, n) for n in names]
    t0 = time.time()
    with get_context('spawn').Pool(nproc, initializer=init, initargs=([0, 1],)) as pool:
        for name, st, sec in pool.imap_unordered(job, jobs):
            print('LIN', tag, name, st, round(sec, 1), round((time.time() - t0) / 60, 1), 'min', flush=True)
    print('LIN_DONE', tag, round((time.time() - t0) / 60, 1), 'min', flush=True)
