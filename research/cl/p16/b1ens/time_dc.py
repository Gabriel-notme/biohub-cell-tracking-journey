"""Extra P-stage time for div_complete fork scoring with an ensemble vs P15's single b1 (fresh embedding caches each time).
usage: time_dc.py <set> <n movies> <ckpt,ckpt,...>   (times dc.score_candidates on the original B5 graphs of the n largest movies)"""
import os, sys, json, time, glob, shutil, tempfile
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'RAYON_NUM_THREADS', 'BLOSC_NTHREADS']: os.environ[k] = '1'
ART = '/workspace/art_b56/artifact_bundle'; os.environ['BIOHUB_ART'] = ART; sys.path.insert(0, ART); sys.path.insert(0, '/workspace/p56stage')
import numcodecs.blosc; numcodecs.blosc.use_threads = False
from pathlib import Path
import torch
from refine_events import EventRefiner
import div_complete as dc
st, n, extra = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(',')
RUN = {'t127a': '/workspace/sync4/runs/b5f_t127a', 't127b': '/workspace/sync3/runs/b5f_t127b', 'hold36': '/workspace/runs/b5f_hold36'}[st]
G = Path(RUN) / 'working/lineage_graphs'; data = '/workspace/cl/data_' + st
names = sorted((p.stem for p in Path(data).glob('*.zarr')), key=lambda x: -(G / (x + '.json')).stat().st_size)[:n]
cfgs = {'P15_b1': ([Path(ART) / 'b1_best.pt'], {'fork_ensemble': 'first', 'edge_ensemble': 'last'}),
        'ens': ([Path(ART) / 'b1_best.pt'] + [Path(p) for p in extra], {'fork_ensemble': 'mean', 'edge_ensemble': 'last'})}
for name in names:
    d = json.loads((G / (name + '.json')).read_text()); nodes = {int(k): v for k, v in d['nodes'].items()}; edges = d['edges']
    line = '%s nodes %d' % (name, len(nodes))
    for lab, (ms, cfg) in cfgs.items():
        tmp = tempfile.mkdtemp(dir='/dev/shm'); er = EventRefiner(ms, data, cfg, tmp)
        torch.cuda.synchronize(); t0 = time.time(); res = dc.score_candidates(er, name, nodes, edges); torch.cuda.synchronize()
        line += ' | %s (%d models) %.2f s, %d cands' % (lab, len(ms), time.time() - t0, len(res)); shutil.rmtree(tmp)
    print(line, flush=True)
