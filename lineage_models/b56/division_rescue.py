"""Long-motion proposal screening followed by detector-aligned seven-frame review."""
from pathlib import Path
import os,json,hashlib
import numpy as np
from scipy.special import logit,expit
from joint_refine import JointRefiner,JOINT_DEFAULT
from motion_refine import MotionRefiner
from refine_events import EventRefiner,structure
from cell_event import chain,fork_geometry
class DivisionRescueRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.root=Path(root);self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir);self.cache.mkdir(exist_ok=True,parents=True);self.motion=MotionRefiner(paths[:1],root,config,cache_dir);self.expert=EventRefiner(paths[1:],root,cache_dir=self.cache.parent/'event_cache');self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    def predict(self,name,nodes,edges):
        self.motion.config=self.config;prior=self.motion.predict(name,nodes,edges);rows=prior['fork'];gate=float(self.config.get('fork_proposal_gate',.02));sha=hashlib.sha256(json.dumps([nodes,edges,gate],sort_keys=True).encode()+rows.tobytes()).hexdigest()[:16];path=self.cache/('division_rescue_'+name+'_'+self.signature+'_'+sha+'.npz')
        if path.exists():
            with np.load(path) as d:cnn=d['cnn_logits'];selected=d['selected']
        else:
            selected=np.flatnonzero(expit(prior['fork_logits'].mean(0))>=gate);cnn=np.full(len(rows),-20,np.float32)
            if len(selected):
                rr=rows[selected];wanted=set(map(int,rr.ravel()));subset={n:v for n,v in nodes.items() if n in wanted};ids,banks=self.expert.embeddings(name,subset);out,prev,_,pos=structure(nodes,edges);geometry=[fork_geometry(chain(int(s),prev,pos),chain(int(a),out,pos),chain(int(b),out,pos)) for s,a,b in rr];p=self.expert.score('fork',[tuple(map(int,r)) for r in rr],geometry,banks,{n:i for i,n in enumerate(ids)});cnn[selected]=logit(np.clip(p,1e-6,1-1e-6))
            tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,cnn_logits=cnn,selected=selected);os.replace(tmp,path)
        result=np.full(len(rows),-20,np.float32);weight=float(self.config.get('cnn_fork_weight',.75));result[selected]=weight*cnn[selected]+(1-weight)*prior['fork_logits'].mean(0)[selected]
        return {**prior,'fork_logits':result[None]}
