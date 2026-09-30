from pathlib import Path
import numpy as np
from joint_refine import JointRefiner,JOINT_DEFAULT
from motion_refine import MotionRefiner
from inverse_dense_refine import BidirectionalDenseRefiner
class HybridMotionRefiner(JointRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        assert len(paths)==5
        self.paths=[Path(p) for p in paths];self.config={**JOINT_DEFAULT,**(config or {})};self.motion=MotionRefiner(paths[:1],root,config,cache_dir);self.dense=BidirectionalDenseRefiner(paths[1:],root,config,cache_dir)
    def predict(self,name,nodes,edges):
        self.motion.config=self.config;self.dense.config=self.config;self.dense.backward.config=self.config
        m=self.motion.predict(name,nodes,edges);d=self.dense.predict(name,nodes,edges);assert np.array_equal(m['edge'],d['edge'])
        alpha=self.config.get('image_blend',.3);logits=(1-alpha)*m['edge_logits'].mean(0)+alpha*d['edge_logits'].mean(0)
        return {**m,'edge_logits':logits[None].astype(np.float32)}
