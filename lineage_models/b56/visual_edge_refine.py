from pathlib import Path
import os,json,hashlib
import numpy as np,torch,lightgbm as lgb
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from event_candidate_refine import EventCandidateRefiner
from fast_motion_features import FastMotionFeatures
from visual_motion_features import visual_edge_features
from candidate_rows import candidate_rows

class VisualEdgeRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==3
        self.paths=[Path(p) for p in paths];self.root=Path(root);self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir);self.cache.mkdir(exist_ok=True,parents=True)
        checkpoint=torch.load(paths[0],map_location='cpu',weights_only=False);assert checkpoint['config']['architecture']=='visual_motion_edge_lgbm';self.models=[lgb.Booster(model_str=s) for s in checkpoint['models']]
        self.appearance=EventCandidateRefiner(paths[1:],root,config,cache_dir);self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    def predict(self,name,nodes,edges):
        self.appearance.config=self.config
        edges_only=self.config['fork_threshold']>1 and self.config['fork_veto']==0 and self.config['preserve_inherited_divisions'] and not self.config.get('fork_repair',False)
        if edges_only:
            er,fr=candidate_rows(nodes,edges,self.config,include_forks=False)
            prior={'edge':er,'fork':fr,'fork_logits':np.empty((1,0),np.float32)}
        else:
            prior=self.appearance.predict(name,nodes,edges)
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()+prior['edge'].tobytes()).hexdigest()[:16];path=self.cache/('visual_edge_sparse_v1_'+name+'_'+self.signature+'_'+graphsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {**prior,'edge_logits':d['edge_logits']}
        rows=prior['edge'];prob=np.empty(len(rows),np.float32)
        if len(rows):
            mf=FastMotionFeatures(name,nodes,edges,self.root,self.cache);ids,banks=self.appearance.expert.embeddings(name,nodes);lookup={n:i for i,n in enumerate(ids)}
            for start in range(0,len(rows),4096):
                rr=rows[start:start+4096];motion=mf.rows('edge',rr);x=visual_edge_features(motion,rr,self.appearance.expert.models,banks,lookup);x[:,[0,1,2,4,5,6,7,8,9,10,11,12]]=0;prob[start:start+len(rr)]=np.mean([m.predict(x,num_threads=2) for m in self.models],axis=0)
        logits=logit(np.clip(prob,1e-6,1-1e-6))[None].astype(np.float32);tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,edge_logits=logits);os.replace(tmp,path)
        return {**prior,'edge_logits':logits}
