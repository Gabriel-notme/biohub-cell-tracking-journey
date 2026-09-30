"""B1/B2 candidate evidence on a common sparse event graph for controlled fusion."""
from pathlib import Path
import hashlib,json,os
import numpy as np
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from refine_events import EventRefiner

class EventCandidateRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.root=Path(root);self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/joint_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
        self.expert=EventRefiner(paths,root,config={'fork_ensemble':'first','edge_ensemble':'last'},cache_dir=self.cache.parent/'event_cache')
    def predict(self,name,nodes,edges):
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        cfgsha=hashlib.sha256(json.dumps({k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']},sort_keys=True).encode()).hexdigest()[:10]
        path=self.cache/('event_'+name+'_'+self.signature+'_'+graphsha+'_'+cfgsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        edge,fork,eg,fg=self.candidates(nodes,edges);ids,emb=self.expert.embeddings(name,nodes);lookup={n:i for i,n in enumerate(ids)}
        ep=self.expert.score('edge',edge,eg,emb,lookup);fp=self.expert.score('fork',fork,fg,emb,lookup)
        result={'edge':np.asarray(edge,np.int64).reshape(-1,2),'fork':np.asarray(fork,np.int64).reshape(-1,3),'edge_logits':logit(np.clip(ep,1e-7,1-1e-7))[None],'fork_logits':logit(np.clip(fp,1e-7,1-1e-7))[None]}
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path);return result
