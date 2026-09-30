from pathlib import Path
import os,json,hashlib
import numpy as np,torch,lightgbm as lgb
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from event_candidate_refine import EventCandidateRefiner
from motion_features import MotionFeatures
from visual_motion_features import visual_fork_features

class VisualMotionRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==3
        self.paths=[Path(p) for p in paths];self.root=Path(root);self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir);self.cache.mkdir(exist_ok=True,parents=True)
        checkpoint=torch.load(paths[0],map_location='cpu',weights_only=False);assert checkpoint['config']['architecture']=='visual_motion_fork_lgbm';self.models=[lgb.Booster(model_str=s) for s in checkpoint['models']]
        self.appearance=EventCandidateRefiner(paths[1:],root,config,cache_dir);self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    def predict(self,name,nodes,edges):
        self.appearance.config=self.config;prior=self.appearance.predict(name,nodes,edges);graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()+prior['fork'].tobytes()).hexdigest()[:16];path=self.cache/('visual_motion_'+name+'_'+self.signature+'_'+graphsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {**prior,'fork_logits':d['fork_logits']}
        rows=prior['fork'];prob=np.empty(len(rows),np.float32)
        if len(rows):
            mf=MotionFeatures(name,nodes,edges,self.root,self.cache);ids,banks=self.appearance.expert.embeddings(name,nodes);lookup={n:i for i,n in enumerate(ids)}
            for start in range(0,len(rows),4096):
                rr=rows[start:start+4096];motion=mf.rows('fork',rr);x=visual_fork_features(motion,rr,self.appearance.expert.models,banks,lookup);prob[start:start+len(rr)]=np.mean([m.predict(x,num_threads=2) for m in self.models],axis=0)
        logits=logit(np.clip(prob,1e-6,1-1e-6))[None].astype(np.float32);tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,fork_logits=logits);os.replace(tmp,path)
        return {**prior,'fork_logits':logits}
