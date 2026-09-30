from pathlib import Path
import os,json,time,hashlib
import numpy as np,torch,lightgbm as lgb
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from motion_features import MotionFeatures,VERSION,invariant_features
from fast_motion_features import FastMotionFeatures
from candidate_rows import candidate_rows

class MotionRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.root=Path(root);self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/joint_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.models=[];self.transforms=[]
        for p in self.paths:
            ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['config']['feature_version']==VERSION
            self.transforms.append(ck['config'].get('rotation_invariant',False))
            self.models.append({k:[lgb.Booster(model_str=s) for s in ss] for k,ss in ck['models'].items()})
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    def predict(self,name,nodes,edges):
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16];cfgsha=hashlib.sha256(json.dumps({k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']},sort_keys=True).encode()).hexdigest()[:10]
        edges_only=self.config['fork_threshold']>1 and self.config['fork_veto']==0 and self.config['preserve_inherited_divisions'] and not self.config.get('fork_repair',False)
        prefix='motion_fast_v3_edges_' if edges_only else 'motion_fast_v2_'
        path=self.cache/(prefix+name+'_'+self.signature+'_'+graphsha+'_'+cfgsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        er,fr=candidate_rows(nodes,edges,self.config,include_forks=not edges_only);features=FastMotionFeatures(name,nodes,edges,self.root,self.cache)
        result={'edge':er,'fork':fr}
        for kind,rows in [('edge',er),('fork',fr)]:
            outputs=[]
            for model,invariant in zip(self.models,self.transforms):
                prob=np.empty(len(rows),np.float32)
                for start in range(0,len(rows),4096):
                    x=features.rows(kind,rows[start:start+4096])
                    if invariant:x=invariant_features(x,kind)
                    prob[start:start+len(x)]=np.mean([m.predict(x,num_threads=2) for m in model[kind]],axis=0)
                outputs.append(logit(np.clip(prob,1e-6,1-1e-6)))
            result[kind+'_logits']=np.asarray(outputs,np.float32)
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path);return result
