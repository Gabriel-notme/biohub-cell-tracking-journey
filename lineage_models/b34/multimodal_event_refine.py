import numpy as np
from joint_refine import JointRefiner,JOINT_DEFAULT
from visual_edge_refine import VisualEdgeRefiner
from visual_motion_refine import VisualMotionRefiner
class MultimodalEventRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==4
        self.config={**JOINT_DEFAULT,**(config or {})};self.edge=VisualEdgeRefiner([paths[0],*paths[2:]],root,config,cache_dir);self.fork=VisualMotionRefiner(paths[1:],root,config,cache_dir)
    def predict(self,name,nodes,edges):
        self.edge.config=self.config;self.fork.config=self.config;a=self.edge.predict(name,nodes,edges);b=self.fork.predict(name,nodes,edges);assert np.array_equal(a['edge'],b['edge']) and np.array_equal(a['fork'],b['fork'])
        return {**a,'fork_logits':b['fork_logits']}
