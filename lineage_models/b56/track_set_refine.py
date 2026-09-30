from pathlib import Path
import os,json,hashlib
import numpy as np,torch
from scipy.special import logit
from track_set import TrackSetNet,mask_incumbent,event_probabilities
from track_video import load_track,TrackMovie,sequences
from track_fusion import encode_bank,fusion_features
from track_video_refine import geometries
from refine_events import structure
from candidate_rows import candidate_rows
from fast_motion_features import FastMotionFeatures
from motion_features import invariant_features
from joint_refine import JointRefiner,JOINT_DEFAULT

class TrackSetRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        torch.backends.mha.set_fastpath_enabled(False);self.paths=[Path(p) for p in paths];self.root=Path(root);self.cache=Path(cache_dir or '/kaggle/working/set_cache');self.cache.mkdir(exist_ok=True,parents=True);self.heads=[];expected=None
        for path in self.paths:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if ck['config'].get('architecture')!='candidate_set_attention':break
            model=TrackSetNet(**ck['config']).cuda().eval();model.load_state_dict(ck['model']);self.heads.append(model)
            if expected is None:expected=ck['encoder_sha256_list']
            else:assert expected==ck['encoder_sha256_list']
        assert self.heads
        encoder_paths=self.paths[len(self.heads):];assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in encoder_paths]==expected
        self.models=[load_track(p)[0] for p in encoder_paths];self.encoder_signatures=[s[:16] for s in expected]
        self.signature=hashlib.sha256(b'candidate-set-v1'+b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16];self.config={**JOINT_DEFAULT,**(config or {})}

    @torch.inference_mode()
    def predict(self,name,nodes,edges):
        cfg={k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']};only_edges=self.config['fork_threshold']>1 and self.config['fork_veto']==0 and self.config['preserve_inherited_divisions'] and not self.config.get('fork_repair',False)
        graph_sig=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16];cfg_sig=hashlib.sha256(json.dumps([cfg,only_edges],sort_keys=True).encode()).hexdigest()[:10];path=self.cache/f'trackset_{name}_{self.signature}_{graph_sig}_{cfg_sig}.npz'
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(ids)};out,prev,frames,pos=structure(nodes,edges);seq,motion=sequences(ids,nodes,out,prev);er,fr=candidate_rows(nodes,edges,self.config,include_forks=not only_edges)
        feature_sig=hashlib.sha256(json.dumps([self.encoder_signatures,graph_sig,cfg['max_distance'],cfg['max_candidates']],sort_keys=True).encode()).hexdigest()[:16];feature_path=self.cache/f'setfeatures_{name}_{feature_sig}.npz'
        if feature_path.exists():
            with np.load(feature_path) as d:assert np.array_equal(d['rows'],er);features=d['features']
        else:
            banks=[]
            for mi,model in enumerate(self.models):
                bankpath=self.cache/f'trackfusion_bank_v3_{name}_{self.encoder_signatures[mi]}_{graph_sig}.npz'
                if bankpath.exists():
                    with np.load(bankpath) as d:bank=d['bank']
                else:
                    movie=TrackMovie(self.root/(name+'.zarr'),model.channels);crops=np.empty((len(ids),*((model.channels,) if model.channels>1 else ()),8,16,16),np.float16)
                    for t,ns in sorted(frames.items()):
                        for start in range(0,len(ns),512):
                            chosen=ns[start:start+512];crops[[lookup[n] for n in chosen]]=movie.crops(t,[[nodes[n][k] for k in ['z','y','x']] for n in chosen])
                    bank=encode_bank(model,crops,seq,motion);tmp=bankpath.with_name(bankpath.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,bank=bank);os.replace(tmp,bankpath)
                banks.append(bank)
            positions=np.asarray([pos[n] for n in ids]);mf=FastMotionFeatures(name,nodes,edges,self.root,self.cache);parts=[]
            for start in range(0,len(er),4096):
                rr=er[start:start+4096];ix=np.asarray([[lookup[int(n)] for n in row] for row in rr]);g=geometries(ix,'edge',positions,seq,motion);m=mf.rows('edge',rr);features=[]
                for model,bank in zip(self.models,banks):features.append(fusion_features(model,bank,ix,invariant_features(m,'edge') if model.edge_dim==338 else g,m,'edge'))
                parts.append(np.concatenate([features[0]]+[f[:,338:] for f in features[1:]],1) if features else invariant_features(m,'edge'))
            features=np.concatenate(parts) if parts else np.empty((0,self.heads[0].input_dim),np.float32);tmp=feature_path.with_name(feature_path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,rows=er,features=features);os.replace(tmp,feature_path)
        result={'edge':er,'fork':fr,'edge_logits':np.empty((len(self.heads),len(er)),np.float32),'fork_logits':np.empty((len(self.heads),len(fr)),np.float32)}
        order=np.argsort(er[:,0],kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(er[order,0]))+1,len(order)];groups=[order[s:t] for s,t in zip(starts[:-1],starts[1:])]
        fork_by_source={}
        for fi,row in enumerate(fr):fork_by_source.setdefault(int(row[0]),[]).append(fi)
        for start in range(0,len(groups),256):
            batch=groups[start:start+256];width=max(map(len,batch));x=np.zeros((len(batch),width,features.shape[1]),np.float32);valid=np.zeros((len(batch),width),bool)
            for j,ix in enumerate(batch):x[j,:len(ix)]=features[ix];valid[j,:len(ix)]=True
            x=mask_incumbent(torch.from_numpy(x).cuda());valid=torch.from_numpy(valid).cuda()
            for mi,model in enumerate(self.heads):
                with torch.autocast('cuda',dtype=torch.float16):outputs=model(x,valid)
                ep,fp=event_probabilities(*outputs,valid,structured=getattr(model,'structured',False));ep=ep.cpu().numpy();fp=fp.cpu().numpy()
                for j,ix in enumerate(batch):
                    result['edge_logits'][mi,ix]=logit(np.clip(ep[j,:len(ix)],1e-6,1-1e-6));target={int(er[q,1]):k for k,q in enumerate(ix)}
                    for fi in fork_by_source.get(int(er[ix[0],0]),[]):
                        aa,bb=sorted([target[int(fr[fi,1])],target[int(fr[fi,2])]]);result['fork_logits'][mi,fi]=logit(np.clip(fp[j,aa,bb],1e-6,1-1e-6))
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path)
        print('SET_PREDICTED',name,len(er),len(fr),flush=True);return result
