import numpy as np
from dense_motion_refine import DenseMotionRefiner

def relative_heat(heat,ids):
    maxima={}
    for n,h in zip(ids,heat):maxima[int(n)]=max(maxima.get(int(n),-1e30),float(h))
    return heat-np.asarray([maxima[int(n)] for n in ids])

class InverseDenseRefiner(DenseMotionRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        super().__init__(paths,root,config,cache_dir);self.reverse_time=True
    def predict(self,name,nodes,edges):
        self.appearance.config=self.config;prior=self.appearance.predict(name,nodes,edges);edge=prior['edge']
        heat=self.correspondence_heat(name,nodes,edge[:,::-1].copy()).mean(0);rel=relative_heat(heat,edge[:,1])
        lp=self.config.get('dense_confidence_bias',4.)+self.config.get('appearance_weight',.25)*np.clip(prior['edge_logits'].mean(0),-10,10)+self.config.get('dense_weight',1.)*rel
        return {**prior,'edge_logits':lp[None].astype(np.float32)}

class BidirectionalDenseRefiner(DenseMotionRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==4
        super().__init__([paths[0],*paths[-2:]],root,config,cache_dir)
        self.backward=InverseDenseRefiner([paths[1],*paths[-2:]],root,config,cache_dir)
    def predict(self,name,nodes,edges):
        self.appearance.config=self.config;prior=self.appearance.predict(name,nodes,edges);edge=prior['edge']
        forward=relative_heat(self.correspondence_heat(name,nodes,edge).mean(0),edge[:,0]);backward=relative_heat(self.backward.correspondence_heat(name,nodes,edge[:,::-1].copy()).mean(0),edge[:,1])
        lp=self.config.get('dense_confidence_bias',4.)+self.config.get('appearance_weight',.25)*np.clip(prior['edge_logits'].mean(0),-10,10)+self.config.get('dense_weight',1.)*forward+self.config.get('inverse_weight',1.)*backward
        return {**prior,'edge_logits':lp[None].astype(np.float32)}
