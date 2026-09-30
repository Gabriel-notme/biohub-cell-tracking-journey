from pathlib import Path
import os,json,multiprocessing as mp
os.environ['OMP_NUM_THREADS']='1';os.environ['POLARS_MAX_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from prepare_visual_motion import select
R=Path('/workspace/biohub');O=R/'ranking_groups'
def run(job):
    name,group=job;target=O/(name+'.npy')
    if target.exists():return name
    _,selected=select(name,group);rows,y=selected['edge']
    with np.load(R/'visual_edge_data'/(name+'.npz')) as d:assert np.array_equal(y,d['edge_y'])
    np.save(target,rows[:,0]);return name
if __name__=='__main__':
    O.mkdir(exist_ok=True);split=json.loads((R/'events/split.json').read_text());jobs=[(n,g) for g in ['train','calibration'] for n in split[g]]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as p:
        for n in p.map(run,jobs):print('RANK_GROUPS',n,flush=True)
    (O/'ready.json').write_text(json.dumps({'movies':len(jobs)}))
