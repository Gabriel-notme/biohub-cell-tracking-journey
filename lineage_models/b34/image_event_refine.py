"""Independent logits from fine-tuned image experts on the common event graph."""
from pathlib import Path
import hashlib,json,os
import numpy as np,torch
from joint_refine import JointRefiner,JOINT_DEFAULT
from refine_events import EventRefiner

class ImageEventRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.root=Path(root);self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/joint_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
        self.expert=EventRefiner(paths,root,cache_dir=self.cache.parent/'event_cache')
    @torch.inference_mode()
    def predict(self,name,nodes,edges):
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16];cfgsha=hashlib.sha256(json.dumps({k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']},sort_keys=True).encode()).hexdigest()[:10]
        path=self.cache/('image_event_'+name+'_'+self.signature+'_'+graphsha+'_'+cfgsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        er,fr,eg,fg=self.candidates(nodes,edges);er=np.asarray(er,np.int64).reshape(-1,2);fr=np.asarray(fr,np.int64).reshape(-1,3);ids,emb=self.expert.embeddings(name,nodes);lookup={n:i for i,n in enumerate(ids)}
        result={'edge':er,'fork':fr}
        for kind,rows,geom in [('edge',er,eg),('fork',fr,fg)]:
            ix=np.asarray([[lookup[int(n)] for n in row] for row in rows],np.int64).reshape(-1,2 if kind=='edge' else 3);outputs=[]
            for model,bank in zip(self.expert.models,emb):
                logits=np.empty(len(rows),np.float32)
                for start in range(0,len(rows),4096):
                    ii=ix[start:start+4096];z=[torch.from_numpy(bank[ii[:,j]]).cuda() for j in range(ii.shape[1])];g=torch.from_numpy(geom[start:start+len(ii)]).cuda()
                    lp=model.edge_logits(*z,g) if kind=='edge' else model.fork_logits(*z,g);logits[start:start+len(ii)]=lp.float().cpu().numpy()
                outputs.append(logits)
            result[kind+'_logits']=np.asarray(outputs,np.float32)
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path);return result
