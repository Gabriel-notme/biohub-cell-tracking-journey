from pathlib import Path
import os,json,time,hashlib
import numpy as np,torch
from joint_refine import JointRefiner,JOINT_DEFAULT
from trajectory_model import load_trajectory,AppearanceMovie,track_nodes,sequence_features
from refine_events import structure

class TrajectoryRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        torch.backends.mha.set_fastpath_enabled(False)
        self.paths=[Path(p) for p in paths];self.models=[load_trajectory(p)[0] for p in paths];self.root=Path(root)
        self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/joint_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def predict(self,name,nodes,edges):
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        cfgsha=hashlib.sha256(json.dumps({k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']},sort_keys=True).encode()).hexdigest()[:10]
        path=self.cache/('trajectory_'+name+'_'+self.signature+'_'+graphsha+'_'+cfgsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        started=time.time();edge,fork,eg,fg=self.candidates(nodes,edges)
        edge=np.asarray(edge,np.int64).reshape(-1,2);fork=np.asarray(fork,np.int64).reshape(-1,3)
        out,prev,frames,pos=structure(nodes,edges)
        # Appearance depends only on image coordinates; reuse across candidate graphs and model versions.
        coordsha=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()).hexdigest()[:16]
        featurepath=self.cache/('appearance_'+name+'_'+coordsha+'.npz')
        ids=sorted(nodes)
        if featurepath.exists():
            with np.load(featurepath) as d:descriptors={int(n):v for n,v in zip(d['ids'],d['features'])}
        else:
            movie=AppearanceMovie(self.root/(name+'.zarr'))
            descriptors=movie.descriptors({n:(nodes[n]['t'],[nodes[n][k] for k in ['z','y','x']]) for n in ids})
            tmp=featurepath.with_name(featurepath.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,ids=np.asarray(ids),features=np.stack([descriptors[n] for n in ids]));os.replace(tmp,featurepath)
        source_ids=sorted(set(edge[:,0].tolist()+fork[:,0].tolist()));target_ids=sorted(set(edge[:,1:].ravel().tolist()+fork[:,1:].ravel().tolist()))
        result={'edge':edge,'fork':fork,'edge_logits':np.empty((len(self.models),len(edge)),np.float32),'fork_logits':np.empty((len(self.models),len(fork)),np.float32)}
        for mi,model in enumerate(self.models):
            embeddings=[];lookups=[]
            for ns,links in [(source_ids,prev),(target_ids,out)]:
                parts=[];lookups.append({n:i for i,n in enumerate(ns)})
                for start in range(0,len(ns),1024):
                    batch=ns[start:start+1024];seqs=[]
                    for n in batch:
                        chain=track_nodes(n,links,nodes);seqs.append(sequence_features([pos[k] for k in chain],descriptors,chain))
                    with torch.autocast('cuda',dtype=torch.float16):z=model.encode(torch.from_numpy(np.stack(seqs)).cuda())
                    parts.append(z)
                embeddings.append(torch.cat(parts))
            for kind,rows,geom in [('edge',edge,eg),('fork',fork,fg)]:
                for start in range(0,len(rows),8192):
                    rr=rows[start:start+8192];zz=[embeddings[0][torch.tensor([lookups[0][int(n)] for n in rr[:,0]],device='cuda')]]
                    zz.extend(embeddings[1][torch.tensor([lookups[1][int(n)] for n in rr[:,j]],device='cuda')] for j in range(1,rr.shape[1]))
                    with torch.autocast('cuda',dtype=torch.float16):p=model.classify_features(torch.stack(zz,1),torch.from_numpy(geom[start:start+len(rr)]).cuda(),kind)
                    result[kind+'_logits'][mi,start:start+len(rr)]=p.float().cpu().numpy()
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path)
        print('TRAJECTORY_INFER',name,'seconds',round(time.time()-started,2),flush=True)
        return result
