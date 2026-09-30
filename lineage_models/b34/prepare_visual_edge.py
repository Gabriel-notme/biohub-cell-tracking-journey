from pathlib import Path
import os,json,time,argparse
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ['BIOHUB_BASE_REPO']='/workspace/biohub/baseline_validation/tracking_repo'
import numpy as np,torch
from prepare_visual_motion import select
from refine_events import EventRefiner
from visual_motion_features import visual_edge_features
R=Path('/workspace/biohub');D=R/'visual_edge_data';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();D.mkdir(exist_ok=True);split=json.loads((R/'events/split.json').read_text());(D/'split.json').write_text(json.dumps(split));torch.set_num_threads(2)
    encoder=EventRefiner([R/'B1/best.pt',R/'B2_pretrained/best.pt'],TRAIN,cache_dir=R/'visual_edge_training_cache')
    jobs=[(n,g) for g in ['calibration','train'] for n in split[g]][a.shard::2]
    for name,group in jobs:
        if (D/(name+'.npz')).exists():continue
        started=time.time();nodes,selected=select(name,group);rows,y=selected['edge']
        with np.load(R/'motion_data'/(name+'.npz')) as d:
            assert np.array_equal(y,d['edge_y']);motion=d['edge_x']
        if len(rows):
            wanted=set(map(int,rows.ravel()));subset={n:v for n,v in nodes.items() if n in wanted};ids,banks=encoder.embeddings(name,subset);x=visual_edge_features(motion,rows,encoder.models,banks,{n:i for i,n in enumerate(ids)})
        else:x=np.empty((0,0),np.float32)
        np.savez_compressed(D/(name+'.npz'),edge_x=x,edge_y=y);print('VISUAL_EDGE_PREPARED',name,len(y),int(y.sum()),round(time.time()-started),flush=True)
    (D/('shard_'+str(a.shard)+'_ready.json')).write_text(json.dumps({'ready':True}))
    if all((D/('shard_'+str(i)+'_ready.json')).exists() for i in range(2)):(D/'ready.json').write_text(json.dumps({'ready':True}))
