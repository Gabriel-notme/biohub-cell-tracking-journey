"""Backward visual correspondence: a detected child predicts its previous parent."""
from pathlib import Path
import os,json,time,multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor,as_completed
from prepare_events import graph
from joint_model import JointMovie,SHAPE
from cell_event import SCALE
R=Path('/workspace/biohub');D=R/'inverse_data';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')

def prepare(name):
    marker=D/(name+'_ready.json')
    if marker.exists():return json.loads(marker.read_text())
    nodes,out,prev,frames,pos=graph(TRAIN/(name+'.geff'));rng=np.random.default_rng(int.from_bytes(name.encode()[-5:],'little')%2**32)
    eligible=[n for n,p in prev.items() if nodes[n]['t']==nodes[p]['t']+1]
    selected=set(rng.choice(eligible,min(384,len(eligible)),replace=False).tolist())
    for s,ds in out.items():
        if len(ds)==2:
            frontier=ds
            for _ in range(3):selected.update(n for n in frontier if n in prev);frontier=[d for n in frontier for d in out.get(n,[])]
    selected=sorted(selected,key=lambda n:(nodes[n]['t'],n));aligned=dict(pos)
    with np.load(R/'training_detector_candidates'/(name+'.npz')) as f:
        for t,ns in frames.items():
            if str(t) not in f or not len(f[str(t)]):continue
            points=f[str(t)]*SCALE;tree=cKDTree(points);pairs=[]
            for n in ns:
                dist,j=tree.query(pos[n])
                if dist<=4.5:pairs.append((dist,n,j))
            used=set()
            for dist,n,j in sorted(pairs):
                if j not in used:aligned[n]=points[j];used.add(j)
    bank=np.lib.format.open_memmap(D/(name+'_patches.npy'),mode='w+',dtype=np.float16,shape=(len(selected),7,*SHAPE));movie=JointMovie(TRAIN/(name+'.zarr'),7);targets=[];weight=[]
    for i,n in enumerate(selected):
        p=prev[n];bank[i]=movie.patch(nodes[n]['t'],aligned[n]/SCALE)[::-1];targets.append(pos[p]-aligned[n]);weight.append(3. if len(out[p])==2 else 1.)
    bank.flush();del bank;np.savez_compressed(D/(name+'.npz'),target=np.asarray(targets,np.float32),weight=np.asarray(weight,np.float32))
    report={'movie':name,'samples':len(selected),'division_samples':sum(w>1 for w in weight)};marker.write_text(json.dumps(report));return report

if __name__=='__main__':
    D.mkdir(exist_ok=True);split=json.loads((R/'events/split.json').read_text());(D/'split.json').write_text(json.dumps(split,indent=2))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(prepare,n) for n in split['calibration']+split['train']]
        for f in as_completed(fs):print('INVERSE_PREPARED',json.dumps(f.result()),flush=True)
    (D/'ready.json').write_text(json.dumps({'movies':len(fs),'time':time.time()}))
