"""Endpoint-only candidate competition with image encoding limited to query tracklets.

All children competing for each queried parent remain in its attention set.
The sparse FP16 batches can differ numerically from dense inference, therefore
this is a separately calibrated deployment implementation, not a bitwise claim.
"""
import hashlib,json,os
import numpy as np,torch
from scipy.special import logit
from track_set_refine import TrackSetRefiner
from endpoint_eligibility import endpoint_only_config,possible_endpoint_pairs
from track_video import TrackMovie,sequences
from track_video_refine import geometries
from track_fusion import fusion_features
from track_set import mask_incumbent,event_probabilities
from candidate_rows import candidate_rows
from refine_events import structure
from fast_motion_features import FastMotionFeatures
from motion_features import invariant_features

class SparseTrackSetRefiner(TrackSetRefiner):
    @torch.inference_mode()
    def predict(self,name,nodes,edges):
        assert endpoint_only_config(self.config),'Sparse implementation supports endpoint-only additions'
        graph_sig=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        cfg_sig=hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest()[:12]
        path=self.cache/f'sparseset_v1_{name}_{self.signature}_{graph_sig}_{cfg_sig}.npz'
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        original=np.asarray([(int(e['source_id']),int(e['target_id'])) for e in edges],np.int64).reshape(-1,2)
        parents={s for s,d in possible_endpoint_pairs(nodes,edges,self.config)}
        er,_=candidate_rows(nodes,edges,self.config,include_forks=False)
        er=er[np.asarray([int(row[0]) in parents for row in er])]
        result={'edge':np.concatenate([original,er]),'fork':np.empty((0,3),np.int64),
                'edge_logits':np.zeros((len(self.heads),len(original)+len(er)),np.float32),
                'fork_logits':np.empty((len(self.heads),0),np.float32)}
        if len(er):
            ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(ids)}
            out,prev,frames,pos=structure(nodes,edges);seq,motion=sequences(ids,nodes,out,prev)
            rows=np.asarray([[lookup[int(n)] for n in row] for row in er]);anchors=[np.unique(rows[:,i]) for i in range(2)]
            needed=np.unique(np.concatenate([seq[i,ix].ravel() for i,ix in enumerate(anchors)]));compact=np.full(len(ids),-1,np.int64);compact[needed]=np.arange(len(needed))
            needed_nodes={ids[int(i)] for i in needed};banks=[]
            for model in self.models:
                movie=TrackMovie(self.root/(name+'.zarr'),model.channels)
                crops=np.empty((len(needed),*((model.channels,) if model.channels>1 else ()),8,16,16),np.float16)
                for t,ns in sorted(frames.items()):
                    chosen=[n for n in ns if n in needed_nodes]
                    for start in range(0,len(chosen),512):
                        part=chosen[start:start+512];crops[[compact[lookup[n]] for n in part]]=movie.crops(t,[[nodes[n][k] for k in ['z','y','x']] for n in part])
                encoded=[]
                for start in range(0,len(crops),512):
                    x=torch.as_tensor(crops[start:start+512].copy(),device='cuda').float()
                    if model.channels==1:x=x[:,None]
                    with torch.autocast('cuda',dtype=torch.float16):encoded.append(model.encode_frame(x))
                encoded=torch.cat(encoded);bank=np.zeros((2,len(ids),model.width),np.float32)
                for direction,indices in enumerate(anchors):
                    for start in range(0,len(indices),1024):
                        ix=indices[start:start+1024];s=torch.as_tensor(compact[seq[direction,ix]],device='cuda');m=torch.as_tensor(motion[direction,ix],device='cuda')
                        assert bool((s>=0).all())
                        with torch.autocast('cuda',dtype=torch.float16):z=model.encode_track(encoded[s],m)
                        bank[direction,ix]=z.float().cpu().numpy()
                banks.append(bank)
            mf=FastMotionFeatures(name,nodes,edges,self.root,self.cache);m=mf.rows('edge',er);g=geometries(rows,'edge',np.asarray([pos[n] for n in ids]),seq,motion)
            parts=[fusion_features(model,bank,rows,invariant_features(m,'edge') if model.edge_dim==338 else g,m,'edge') for model,bank in zip(self.models,banks)]
            features=np.concatenate([parts[0]]+[p[:,338:] for p in parts[1:]],1)
            order=np.argsort(er[:,0],kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(er[order,0]))+1,len(order)];groups=[order[a:b] for a,b in zip(starts[:-1],starts[1:])]
            for start in range(0,len(groups),256):
                batch=groups[start:start+256];width=max(map(len,batch));x=np.zeros((len(batch),width,features.shape[1]),np.float32);valid=np.zeros((len(batch),width),bool)
                for j,ix in enumerate(batch):x[j,:len(ix)]=features[ix];valid[j,:len(ix)]=True
                x=mask_incumbent(torch.from_numpy(x).cuda());valid=torch.from_numpy(valid).cuda()
                for mi,model in enumerate(self.heads):
                    with torch.autocast('cuda',dtype=torch.float16):outputs=model(x,valid)
                    ep,_=event_probabilities(*outputs,valid,structured=getattr(model,'structured',False));ep=ep.cpu().numpy()
                    for j,ix in enumerate(batch):result['edge_logits'][mi,len(original)+ix]=logit(np.clip(ep[j,:len(ix)],1e-6,1-1e-6))
            print('SPARSE_SET_ENCODED',name,'parents',len(parents),'crops',len(needed),'of',len(ids),flush=True)
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path);return result
