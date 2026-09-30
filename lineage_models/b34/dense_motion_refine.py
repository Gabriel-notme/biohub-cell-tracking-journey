"""Parent-conditioned spatial correspondence likelihoods for global association."""
from pathlib import Path
import os,json,hashlib
import numpy as np,torch
from torch.nn import functional as F
from scipy.special import expit,logit
from joint_refine import JointRefiner
from event_candidate_refine import EventCandidateRefiner
from joint_model import JointMovie,SHAPE,SPACING
from cell_event import SCALE

class DenseMotionRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        # The final two checkpoints are unchanged B1/B2 appearance references;
        # preceding checkpoints form a dense correspondence ensemble.
        assert len(paths)>=3
        super().__init__(paths[:-2],root,config,cache_dir)
        self.appearance=EventCandidateRefiner(paths[-2:],root,config,cache_dir)
        self.all_paths=[Path(p) for p in paths]
    @torch.inference_mode()
    def correspondence_heat(self,name,nodes,edge):
        digest=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()+edge.tobytes()).hexdigest()[:16];prefix='inverse_heat_v1_' if getattr(self,'reverse_time',False) else 'dense_heat_v2_';path=self.cache/(prefix+name+'_'+self.signature+'_'+digest+'.npz')
        if path.exists():
            with np.load(path) as d:return d['heat']
        bysource={}
        for i,(s,d) in enumerate(edge):bysource.setdefault(int(s),[]).append(i)
        # Keep reference batches intact to preserve convolution numerics; only
        # unused point-feature pooling and classifier heads are eliminated.
        sources=sorted(bysource,key=lambda s:(nodes[s]['t'],s));heat=np.zeros((len(self.models),len(edge)),np.float32);movie=JointMovie(self.root/(name+'.zarr'),max(m.context for m in self.models))
        for start in range(0,len(sources),32):
            batch=sources[start:start+32];x=torch.from_numpy(movie.patches([nodes[s]['t'] for s in batch],[[nodes[s][k] for k in ['z','y','x']] for s in batch])).cuda().float();maxq=max(len(bysource[s]) for s in batch);coords=np.zeros((len(batch),maxq,3),np.float32)
            for bi,s in enumerate(batch):
                for qi,ix in enumerate(bysource[s]):coords[bi,qi]=[(nodes[int(edge[ix,1])][k]-nodes[s][k])*float(scale) for k,scale in zip(['z','y','x'],SCALE)]
            if getattr(self,'reverse_time',False):x=x.flip(1)
            qt=torch.from_numpy(coords).cuda();shape=qt.new_tensor(SHAPE);spacing=qt.new_tensor(SPACING);grid=((qt/spacing+shape.floor_divide(2))/(shape-1)*2-1).flip(-1).reshape(len(qt),-1,1,1,3)
            for mi,model in enumerate(self.models):
                with torch.autocast('cuda',dtype=torch.float16):encoded=model.encode(x)
                hm=F.max_pool3d(encoded[-2].float(),kernel_size=(3,5,5),stride=1,padding=(1,2,2));sample=F.grid_sample(hm,grid,mode='bilinear',padding_mode='border',align_corners=True)[:,0,:,0,0].cpu().numpy()
                for bi,s in enumerate(batch):heat[mi,bysource[s]]=sample[bi,:len(bysource[s])]
            if start%2048==0:print('DENSE_HEAT_ONLY',name,start,'/',len(sources),'of',len(bysource),'sources',flush=True)
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,heat=heat);os.replace(tmp,path);return heat
    def predict(self,name,nodes,edges):
        self.appearance.config=self.config;prior=self.appearance.predict(name,nodes,edges)
        heat=self.correspondence_heat(name,nodes,prior['edge']).mean(0);sources=prior['edge'][:,0]
        maxh={}
        for s,h in zip(sources,heat):maxh[int(s)]=max(maxh.get(int(s),-1e30),float(h))
        relative=heat-np.asarray([maxh[int(s)] for s in sources])
        logits=self.config.get('dense_confidence_bias',4.)+self.config.get('appearance_weight',.25)*np.clip(prior['edge_logits'].mean(0),-10,10)+self.config.get('dense_weight',1.)*relative
        return {'edge':prior['edge'],'fork':prior['fork'],'edge_logits':logits[None].astype(np.float32),'fork_logits':prior['fork_logits']}
