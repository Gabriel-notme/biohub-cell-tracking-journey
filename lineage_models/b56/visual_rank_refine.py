from pathlib import Path
import os,json,hashlib
import numpy as np,torch,lightgbm as lgb
from joint_refine import JointRefiner,JOINT_DEFAULT
from motion_refine import MotionRefiner
from refine_events import EventRefiner
from fast_motion_features import FastMotionFeatures
from visual_motion_features import visual_edge_features
class VisualRankRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==4
        self.paths=[Path(p) for p in paths];self.root=Path(root);self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir);self.cache.mkdir(exist_ok=True,parents=True);self.motion=MotionRefiner(paths[:1],root,config,cache_dir);self.expert=EventRefiner(paths[2:],root,cache_dir=self.cache.parent/'event_cache')
        ck=torch.load(paths[1],map_location='cpu',weights_only=False);assert ck['config']['architecture']=='visual_edge_ranker';self.models=[lgb.Booster(model_str=s) for s in ck['models']];self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    def predict(self,name,nodes,edges):
        self.motion.config=self.config;prior=self.motion.predict(name,nodes,edges);rows=prior['edge'];sha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()+rows.tobytes()).hexdigest()[:16];path=self.cache/('visual_rank_'+name+'_'+self.signature+'_'+sha+'.npz')
        if path.exists():
            with np.load(path) as d:ranks=d['ranks']
        else:
            mf=FastMotionFeatures(name,nodes,edges,self.root,self.cache);ids,banks=self.expert.embeddings(name,nodes);lookup={n:i for i,n in enumerate(ids)};ranks=np.empty(len(rows),np.float32)
            for start in range(0,len(rows),4096):
                rr=rows[start:start+4096];x=visual_edge_features(mf.rows('edge',rr),rr,self.expert.models,banks,lookup);x[:,[0,1,2,4,5,6,7,8,9,10,11,12]]=0;ranks[start:start+len(rr)]=np.mean([m.predict(x,num_threads=2) for m in self.models],0)
            tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,ranks=ranks);os.replace(tmp,path)
        maxima={}
        for row,p in zip(rows,ranks):maxima[int(row[0])]=max(maxima.get(int(row[0]),-1e30),float(p))
        relative=ranks-np.asarray([maxima[int(row[0])] for row in rows]);lp=prior['edge_logits'].mean(0)+self.config.get('rank_weight',1.)*relative
        return {**prior,'edge_logits':lp[None].astype(np.float32)}
