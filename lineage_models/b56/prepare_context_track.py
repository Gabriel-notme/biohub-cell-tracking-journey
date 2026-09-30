from pathlib import Path
import os,json,hashlib,argparse,time
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,torch
from context_track import ContextTrackNet
from cell_event import Movie
from fast_motion_features import FastMotionFeatures
from motion_features import invariant_features
R=Path('/workspace/biohub');D=R/'track_b4w_3';O=R/'track_context7';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();O.mkdir(exist_ok=True);torch.set_num_threads(3)
    split=json.loads((D/'split.json').read_text());ck=torch.load(R/'centroid_v1_frozen.pt',map_location='cpu',weights_only=False)
    config={'architecture':'context_track','base_config':ck['config']['base_config'],'width':64,'channels':7,'edge_dim':338,'fork_dim':530}
    model=ContextTrackNet(**config).cuda().eval();base={k[len('base.'):]:v for k,v in ck['model'].items() if k.startswith('base.')};model.frame.load_state_dict(base)
    jobs=[(n,g) for g in ['train','calibration'] for n in split[g]][a.shard::2]
    for name,group in jobs:
        if (O/(name+'.npz')).exists():continue
        raw=json.loads((R/('b4_training_graphs' if group=='train' else 'reproduced_B4')/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()}
        with np.load(D/(name+'.npz')) as d:sample={k:d[k] for k in d.files}
        movie=Movie(TRAIN/(name+'.zarr'),7);ids=sample['ids'];order=sorted(range(len(ids)),key=lambda i:nodes[int(ids[i])]['t']);latent=np.empty((len(ids),64),np.float16)
        with torch.inference_mode():
            for start in range(0,len(order),192):
                ii=order[start:start+192];chosen=[nodes[int(ids[i])] for i in ii]
                x=torch.from_numpy(movie.patches([n['t'] for n in chosen],[[n[k] for k in ['z','y','x']] for n in chosen])).cuda().float()
                with torch.autocast('cuda',dtype=torch.float16):z=model.encode_frame(x)
                latent[ii]=z.cpu().numpy()
        mf=FastMotionFeatures(name,nodes,raw['edges'],TRAIN,R/'track_fusion_motion_cache')
        for kind in ['edge','fork']:sample[kind+'_geom']=invariant_features(mf.rows(kind,ids[sample[kind][:,:-1]]),kind)
        if group=='train':
            with np.load(R/'track_residual3'/(name+'.npz')) as d:
                for kind in ['edge','fork']:sample[kind+'_weight']=d[kind+'_weight']
        np.save(O/(name+'_latent.npy'),latent);np.savez_compressed(O/(name+'.npz'),**sample);print('CONTEXT7_BANK',name,len(ids),flush=True)
    (O/('ready_'+str(a.shard)+'.json')).write_text(json.dumps({'movies':len(jobs)}))
    if a.shard==0:
        while not (O/'ready_1.json').exists():time.sleep(5)
        (O/'split.json').write_text(json.dumps(split));(O/'config.json').write_text(json.dumps(config))
        torch.save({'config':config,'model':model.state_dict()},O/'initial.pt')
        (O/'ready.json').write_text(json.dumps({'movies':len(split['train'])+len(split['calibration']),'inherited_encoder_sha256':hashlib.sha256((R/'centroid_v1_frozen.pt').read_bytes()).hexdigest()}));print('CONTEXT7_READY',flush=True)
