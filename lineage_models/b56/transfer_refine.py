from pathlib import Path
import os,json,time,hashlib
import numpy as np,torch
from joint_refine import JointRefiner,JOINT_DEFAULT
from transfer_model import load_transfer
from cell_event import Movie

class TransferRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False)
        self.paths=[Path(p) for p in paths];self.models=[load_transfer(p)[0] for p in paths];self.root=Path(root)
        self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/joint_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
        hashes=[hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in sorted(m.base.state_dict().items()))).hexdigest()[:16] for m in self.models]
        assert len(set(hashes))==1,'Feature cache requires a common frozen visual encoder';self.base_signature=hashes[0]
    @torch.inference_mode()
    def predict(self,name,nodes,edges):
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        cfgsha=hashlib.sha256(json.dumps({k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']},sort_keys=True).encode()).hexdigest()[:10]
        path=self.cache/('transfer_'+name+'_'+self.signature+'_'+graphsha+'_'+cfgsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        started=time.time();edge,fork,eg,fg=self.candidates(nodes,edges);edge=np.asarray(edge,np.int64).reshape(-1,2);fork=np.asarray(fork,np.int64).reshape(-1,3)
        ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(ids)}
        coordsha=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()).hexdigest()[:16];featurepath=self.cache/('transfer_features_'+name+'_'+self.base_signature+'_'+coordsha+'.npz')
        if featurepath.exists():
            with np.load(featurepath) as d:bank=d['features'];assert np.array_equal(d['ids'],ids)
        else:
            movie=Movie(self.root/(name+'.zarr'),context=7);bank=np.empty((len(ids),3,64),np.float16)
            for start in range(0,len(ids),192):
                ns=ids[start:start+192];x=torch.from_numpy(movie.patches([nodes[n]['t'] for n in ns],[[nodes[n][k] for k in ['z','y','x']] for n in ns])).cuda().float()
                with torch.autocast('cuda',dtype=torch.float16):z=self.models[0].base.encode(torch.cat([x[:,0:3],x[:,2:5],x[:,4:7]],0))
                bank[start:start+len(ns)]=z.reshape(3,len(ns),64).transpose(0,1).float().cpu().numpy()
            tmp=featurepath.with_name(featurepath.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,ids=np.asarray(ids),features=bank);os.replace(tmp,featurepath)
        result={'edge':edge,'fork':fork,'edge_logits':np.empty((len(self.models),len(edge)),np.float32),'fork_logits':np.empty((len(self.models),len(fork)),np.float32)}
        xt=torch.from_numpy(bank).cuda().float();central=xt[:,1]
        for mi,m in enumerate(self.models):
            z=[]
            for start in range(0,len(xt),2048):
                with torch.autocast('cuda',dtype=torch.float16):z.append(m.encode(xt[start:start+2048]))
            z=torch.cat(z)
            for kind,rows,geom in [('edge',edge,eg),('fork',fork,fg)]:
                for start in range(0,len(rows),8192):
                    rr=rows[start:start+8192];ix=torch.tensor([[lookup[int(n)] for n in r] for r in rr],device='cuda')
                    with torch.autocast('cuda',dtype=torch.float16):p=m.classify_features(z[ix],central[ix],torch.from_numpy(geom[start:start+len(rr)]).cuda(),kind)
                    result[kind+'_logits'][mi,start:start+len(rr)]=p.float().cpu().numpy()
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path);print('TRANSFER_INFER',name,'seconds',round(time.time()-started,2),flush=True);return result
