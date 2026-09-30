"""Image-tracklet embeddings plus invariant physical motion, with learned event heads."""
from pathlib import Path
import os,json,hashlib,zipfile
import numpy as np,torch,lightgbm as lgb
from scipy.special import logit
from track_video import load_track,TrackMovie,sequences
from track_video_refine import geometries
from refine_events import structure
from candidate_rows import candidate_rows
from fast_motion_features import FastMotionFeatures
from motion_features import invariant_features
from joint_refine import JointRefiner,JOINT_DEFAULT

def drop_incumbent_features(x,kind):
    x=x.copy();offsets=[0] if kind=='edge' else [530-96,530-48]
    for offset in offsets:x[:,offset+44]=.5;x[:,offset+45]=0.
    return x

@torch.inference_mode()
def encode_bank(model,crops,seq,motion):
    parts=[]
    for start in range(0,len(crops),512):
        x=torch.as_tensor(np.asarray(crops[start:start+512]).copy(),device='cuda').float()
        if model.channels==1:x=x[:,None]
        with torch.autocast('cuda',dtype=torch.float16):parts.append(model.encode_frame(x))
    f=torch.cat(parts);emb=[]
    for direction in range(2):
        parts=[]
        for start in range(0,len(crops),1024):
            ix=torch.as_tensor(seq[direction,start:start+1024],device='cuda');m=torch.as_tensor(motion[direction,start:start+1024],device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):parts.append(model.encode_track(f[ix],m).float().cpu().numpy())
        emb.append(np.concatenate(parts))
    return np.stack(emb)

@torch.inference_mode()
def fusion_features(model,bank,rows,geometry,motion,kind):
    if not len(rows):return np.empty((0,motion.shape[1]+(259 if kind=='edge' else 260)),np.float32)
    parts=[]
    for start in range(0,len(rows),4096):
        rr=rows[start:start+4096];z=torch.as_tensor(np.stack([bank[int(j>0),rr[:,j]] for j in range(rr.shape[1])],1),device='cuda')
        g=torch.as_tensor(geometry[start:start+len(rr)],device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model.classify(z,g,kind);phase=model.phase(z).squeeze(-1)
        p,a=z[:,0],z[:,1]
        if kind=='edge':feature=torch.cat([logits[:,None],phase,p,a,(p-a).abs(),p*a],1)
        else:
            b=z[:,2];feature=torch.cat([logits[:,None],phase[:,0:1],phase[:,1:].amin(1,keepdim=True),phase[:,1:].amax(1,keepdim=True),p,(a+b)/2,(a-b).abs(),a*b],1)
        parts.append(feature.float().cpu().numpy())
    image=np.concatenate(parts);motion=invariant_features(motion,kind)
    return np.nan_to_num(np.c_[motion,image]).astype(np.float32)

class TrackFusionRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];assert len(paths)>=2
        checkpoint=torch.load(paths[0],map_location='cpu',weights_only=False)
        assert checkpoint['config']['architecture'] in ['track_video_motion_fusion','track_video_motion_dual_fusion']
        self.heads={k:[lgb.Booster(model_str=s) for s in v] for k,v in checkpoint['models'].items()}
        self.drop_incumbent=checkpoint['config'].get('drop_incumbent',False)
        self.models=[load_track(p)[0] for p in paths[1:]];self.model=self.models[0];self.root=Path(root)
        self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b'fusion-members-v2'+b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
        expected=checkpoint.get('encoder_sha256_list',[checkpoint.get('encoder_sha256')])
        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths[1:]]==expected,'Encoder/head mismatch'
        self.encoder_signatures=[x[:16] for x in expected]
    def predict(self,name,nodes,edges):
        cfg={k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']}
        edges_only=self.config['fork_threshold']>1 and self.config['fork_veto']==0 and self.config['preserve_inherited_divisions'] and not self.config.get('fork_repair',False) and not self.config.get('compute_unused_forks',False)
        if edges_only:cfg['include_forks']=False
        sig=hashlib.sha256(json.dumps([nodes,edges,cfg],sort_keys=True).encode()).hexdigest()[:16]
        path=self.cache/f'trackfusion_{name}_{self.signature}_{sig}.npz'
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(ids)}
        out,prev,frames,pos=structure(nodes,edges);seq,motion=sequences(ids,nodes,out,prev)
        banks=[]
        graph_sig=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        for mi,model in enumerate(self.models):
            bankpath=self.cache/f'trackfusion_bank_v3_{name}_{self.encoder_signatures[mi]}_{graph_sig}.npz'
            legacy=self.cache/f'trackfusion_bank_{name}_{self.signature}_{sig}_{mi}.npz';bank=None;canonical_valid=False
            for candidate in [bankpath,legacy]:
                if not candidate.exists():continue
                try:
                    with np.load(candidate) as d:bank=d['bank']
                    assert bank.shape==(2,len(ids),model.width)
                    canonical_valid=candidate==bankpath
                    break
                except (EOFError,ValueError,zipfile.BadZipFile):
                    bank=None
            if bank is None:
                channels=model.channels;movie=TrackMovie(self.root/(name+'.zarr'),channels)
                crops=np.empty((len(ids),*((channels,) if channels>1 else ()),8,16,16),np.float16)
                for t,ns in sorted(frames.items()):
                    for start in range(0,len(ns),512):
                        chosen=ns[start:start+512];crops[[lookup[n] for n in chosen]]=movie.crops(t,[[nodes[n][k] for k in ['z','y','x']] for n in chosen])
                bank=encode_bank(model,crops,seq,motion)
            if not canonical_valid:
                temporary=bankpath.with_name(bankpath.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(temporary,bank=bank);os.replace(temporary,bankpath)
            banks.append(bank)
        er,fr=candidate_rows(nodes,edges,self.config,include_forks=not edges_only);result={'edge':er,'fork':fr};mf=FastMotionFeatures(name,nodes,edges,self.root,self.cache)
        positions=np.asarray([pos[n] for n in ids])
        for kind,rows in [('edge',er),('fork',fr)]:
            probability=[];member_probability=[]
            for start in range(0,len(rows),4096):
                rr=rows[start:start+4096];ix=np.asarray([[lookup[int(n)] for n in row] for row in rr])
                g=geometries(ix,kind,positions,seq,motion);m=mf.rows(kind,rr)
                parts=[fusion_features(model,bank,ix,g,m,kind) for model,bank in zip(self.models,banks)]
                x=np.concatenate([parts[0]]+[part[:,m.shape[1]:] for part in parts[1:]],1)
                if self.drop_incumbent:x=drop_incumbent_features(x,kind)
                members=np.asarray([head.predict(x,num_threads=2) for head in self.heads[kind]])
                probability.append(members.mean(0));member_probability.append(members)
            q=np.concatenate(probability) if probability else np.empty(0)
            result[kind+'_logits']=logit(np.clip(q,1e-6,1-1e-6))[None].astype(np.float32)
            if kind=='edge':
                members=np.concatenate(member_probability,1) if member_probability else np.empty((len(self.heads[kind]),0))
                result['edge_member_logits']=logit(np.clip(members,1e-6,1-1e-6)).astype(np.float32)
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path)
        return result
