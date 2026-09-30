from pathlib import Path
import os,json,time,hashlib
import numpy as np,torch
from track_video import load_track,TrackMovie,sequences
from cell_event import SCALE
from refine_events import structure
from candidate_rows import candidate_rows
from joint_refine import JointRefiner,JOINT_DEFAULT

def geometries(rows,kind,pos,seq,motion):
    if not len(rows):return np.empty((0,16 if kind=='edge' else 28),np.float32)
    s=rows[:,0];a=rows[:,1];h=pos[seq[0,s,:4]];aa=pos[seq[1,a,:4]]
    nh=motion[0,s,:4,3].sum(1);na=motion[1,a,:4,3].sum(1)
    p=h[:,0];q=aa[:,0];v=(p-h[:,-1])/np.maximum(1,nh-1)[:,None]
    w=(aa[:,-1]-q)/np.maximum(1,na-1)[:,None];d=q-p
    norm=lambda x:np.linalg.norm(x,axis=-1)
    if kind=='edge':return np.c_[d/10,norm(d)/10,v/10,(d-v)/10,w/10,norm(w-v)/10,nh>1,na>1].astype(np.float32)
    b=rows[:,2];bb=pos[seq[1,b,:4]];nb=motion[1,b,:4,3].sum(1);db=bb[:,0]-p
    da=norm(d);dd=norm(db);r=[np.minimum(da,dd)/10,np.maximum(da,dd)/10,norm(q-bb[:,0])/10,
        (d*db).sum(1)/(da*dd+1e-4),norm((q+bb[:,0])/2-p-v)/10,np.abs(da-dd)/10,norm(v)/10]
    for k in range(4):
        qa=aa[:,k];qb=bb[:,k];r.extend([norm(qa-qb)/10,norm((qa+qb)/2-p-(k+1)*v)/10,np.minimum(norm(qa-p),norm(qb-p))/10,np.maximum(norm(qa-p),norm(qb-p))/10])
    vb=(bb[:,-1]-bb[:,0])/np.maximum(1,nb-1)[:,None]
    r.extend([norm(w-vb)/10,(w*vb).sum(1)/(norm(w)*norm(vb)+1e-4),nh/4,np.minimum(na,nb)/4,np.maximum(na,nb)/4])
    return np.stack(r,1).astype(np.float32)

class TrackVideoRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        torch.backends.mha.set_fastpath_enabled(False)
        self.paths=[Path(p) for p in paths];self.models=[load_track(p)[0] for p in paths];self.root=Path(root)
        assert len({(m.channels,m.edge_dim,m.fork_dim) for m in self.models})==1
        self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/joint_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def predict(self,name,nodes,edges):
        started=time.time();graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        candidate_config={k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']}
        edges_only=self.config['fork_threshold']>1 and self.config['fork_veto']==0 and self.config['preserve_inherited_divisions'] and not self.config.get('fork_repair',False) and not self.config.get('compute_unused_forks',False)
        if edges_only:candidate_config['include_forks']=False
        cfgsha=hashlib.sha256(json.dumps(candidate_config,sort_keys=True).encode()).hexdigest()[:10]
        path=self.cache/('trackvideo_'+name+'_'+self.signature+'_'+graphsha+'_'+cfgsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(ids)}
        out,prev,frames,posdict=structure(nodes,edges);seq,motion=sequences(ids,nodes,out,prev)
        pos=np.asarray([posdict[n] for n in ids]);er,fr=candidate_rows(nodes,edges,self.config,include_forks=not edges_only)
        indices={kind:np.asarray([[lookup[int(n)] for n in row] for row in rows],np.int64).reshape(-1,2 if kind=='edge' else 3) for kind,rows in [('edge',er),('fork',fr)]}
        if self.models[0].edge_dim==338:
            from fast_motion_features import FastMotionFeatures
            from motion_features import invariant_features
            mf=FastMotionFeatures(name,nodes,edges,self.root,self.cache)
            geom={k:invariant_features(mf.rows(k,rows),k) for k,rows in [('edge',er),('fork',fr)]}
        else:geom={k:geometries(rr,k,pos,seq,motion) for k,rr in indices.items()}
        coordsha=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()).hexdigest()[:16]
        featurepath=self.cache/('trackframes_'+name+'_'+coordsha+'_'+self.signature+'.npz')
        if featurepath.exists():
            with np.load(featurepath) as d:features=[d['m'+str(i)] for i in range(len(self.models))]
        else:
            channels=self.models[0].channels;assert all(m.channels==channels for m in self.models)
            wide_context=getattr(self.models[0],'patch_shape',None)==(12,24,24)
            if wide_context:
                from cell_event import Movie
                movie=Movie(self.root/(name+'.zarr'),channels)
            else:movie=TrackMovie(self.root/(name+'.zarr'),channels)
            features=[np.empty((len(ids),m.width),np.float32) for m in self.models]
            for t,ns in sorted(frames.items()):
                for start in range(0,len(ns),384):
                    chosen=ns[start:start+384];ix=[lookup[n] for n in chosen]
                    coords=[[nodes[n][k] for k in ['z','y','x']] for n in chosen]
                    crops=movie.patches([t]*len(chosen),coords) if wide_context else movie.crops(t,coords)
                    x=torch.from_numpy(crops).cuda().float()
                    if channels==1:x=x.unsqueeze(1)
                    for mi,model in enumerate(self.models):
                        with torch.autocast('cuda',dtype=torch.float16):z=model.encode_frame(x)
                        features[mi][ix]=z.float().cpu().numpy()
            tmp=featurepath.with_name(featurepath.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**{'m'+str(i):f for i,f in enumerate(features)});os.replace(tmp,featurepath)
        result={'edge':er,'fork':fr}
        for kind in ['edge','fork']:result[kind+'_logits']=np.empty((len(self.models),len(indices[kind])),np.float32)
        for mi,(model,bank) in enumerate(zip(self.models,features)):
            f=torch.from_numpy(bank).cuda();emb=[]
            for direction in range(2):
                parts=[]
                for start in range(0,len(ids),1024):
                    i=torch.from_numpy(seq[direction,start:start+1024]).cuda();m=torch.from_numpy(motion[direction,start:start+1024]).cuda()
                    with torch.autocast('cuda',dtype=torch.float16):z=model.encode_track(f[i],m)
                    parts.append(z)
                emb.append(torch.cat(parts))
            for kind,rr in indices.items():
                for start in range(0,len(rr),8192):
                    ix=torch.from_numpy(rr[start:start+8192]).cuda();z=torch.stack([emb[int(j>0)][ix[:,j]] for j in range(ix.shape[1])],1)
                    g=torch.from_numpy(geom[kind][start:start+len(ix)]).cuda()
                    with torch.autocast('cuda',dtype=torch.float16):logits=model.classify(z,g,kind)
                    result[kind+'_logits'][mi,start:start+len(ix)]=logits.float().cpu().numpy()
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path)
        print('TRACK_VIDEO_INFER',name,len(er),len(fr),round(time.time()-started,2),flush=True);return result
