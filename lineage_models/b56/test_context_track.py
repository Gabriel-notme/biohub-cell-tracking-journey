from pathlib import Path
import numpy as np,torch,json
from context_track import ContextTrackNet
from cell_event import Movie
from train_context_track import forward
R=Path('/workspace/biohub');D=R/'track_context7';TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(843)
name=next(p.stem for p in D.glob('*.npz'));ck=torch.load(R/'centroid_v1_frozen.pt',map_location='cpu',weights_only=False)
model=ContextTrackNet(base_config=ck['config']['base_config'],channels=7,width=64,edge_dim=338,fork_dim=530).cuda().eval();model.frame.load_state_dict({k[5:]:v for k,v in ck['model'].items() if k.startswith('base.')})
with np.load(D/(name+'.npz')) as d:sample={k:d[k] for k in d.files}
latent=np.load(D/(name+'_latent.npy'));bank=torch.from_numpy(latent[sample['seq']]).cuda();moves=torch.from_numpy(sample['motion']).cuda()
raw=json.loads((R/'b4_training_graphs'/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()};movie=Movie(TRAIN/(name+'.zarr'),7);report={}
with torch.inference_mode():
    for kind in ['edge','fork']:
        rows=sample[kind][:8];geometry=torch.from_numpy(sample[kind+'_geom'][:8]).cuda();rr=rows[:,:-1]
        p=forward(model,bank,moves,torch.from_numpy(rows).cuda(),geometry,kind)
        inds=np.stack([sample['seq'][int(j>0),rr[:,j]] for j in range(rr.shape[1])],1);ids=sample['ids'][inds];chosen=[nodes[int(n)] for n in ids.ravel()]
        x=torch.from_numpy(movie.patches([n['t'] for n in chosen],[[n[k] for k in ['z','y','x']] for n in chosen])).cuda().float()
        m=torch.stack([moves[int(j>0),torch.from_numpy(rr[:,j]).cuda()] for j in range(rr.shape[1])],1)
        b,k,t=inds.shape
        with torch.autocast('cuda',dtype=torch.float16):
            z=model.encode_frame(x).reshape(b*k,t,64);z=model.encode_track(z,m.reshape(b*k,t,4)).reshape(b,k,64);direct=model.classify(z,geometry,kind)
            if kind=='fork':swapped=model.classify(z[:,[0,2,1]],geometry,kind);assert torch.equal(direct,swapped)
        difference=float((p-direct.float()).abs().max());assert difference<.015,difference
        report[kind]={'cached_vs_raw_logit_max_error':difference,'examples':len(rows)}
report['seven_frame_spatial_context_verified']=True;report['daughter_order_invariant']=True
(R/'context_track_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
