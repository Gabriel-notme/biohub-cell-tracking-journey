from pathlib import Path
import os,json,time,hashlib,argparse
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,torch
from track_video import load_track
from track_fusion import encode_bank,fusion_features
from fast_motion_features import FastMotionFeatures
from motion_features import invariant_features
R=Path('/workspace/biohub');D=R/'track_corrupt3';O=R/'TrackSet3_features';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();O.mkdir(exist_ok=True);torch.set_num_threads(3)
    encoder=R/'TrackRobust_137/best.pt';model,ck=load_track(encoder);split=json.loads((D/'split.json').read_text());names=[(n,g) for g in ['train','calibration'] for n in split[g]][a.shard::2]
    for name,group in names:
        dest=O/(name+'.npz')
        if dest.exists():continue
        actual=split.get('augmentation_source_movies',{}).get(name,name)
        graph_folder='corrupted_training_graphs' if name.startswith('aug__') else ('b4_training_graphs' if group=='train' else 'reproduced_B4')
        raw=json.loads((R/graph_folder/(actual+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()}
        with np.load(D/(name+'.npz')) as d:sample={k:d[k] for k in d.files}
        crops=np.load(D/(name+'_crops.npy'),mmap_mode='r');bank=encode_bank(model,crops,sample['seq'],sample['motion']);rows=sample['edge'];rr=sample['ids'][rows[:,:-1]]
        mf=FastMotionFeatures(actual,nodes,raw['edges'],TRAIN,R/'set_motion_cache');motion=mf.rows('edge',rr)
        geometry=invariant_features(motion,'edge') if model.edge_dim==338 else sample['edge_geom']
        features=fusion_features(model,bank,rows[:,:-1],geometry,motion,'edge')
        temp=dest.with_name(dest.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(temp,edge=features,edge_y=rows[:,-1],source=rows[:,0]);os.replace(temp,dest)
        print('SET_FEATURES',name,len(rows),flush=True)
    (O/('ready_'+str(a.shard)+'.json')).write_text(json.dumps({'movies':len(names)}))
    if a.shard==0:
        while not (O/'ready_1.json').exists():time.sleep(5)
        (O/'split.json').write_text(json.dumps(split));(O/'ready.json').write_text(json.dumps({'encoder':'TrackRobust_137/best.pt','encoder_sha256':hashlib.sha256(encoder.read_bytes()).hexdigest()}))
