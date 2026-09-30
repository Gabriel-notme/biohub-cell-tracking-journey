import numpy as np
from joint_refine import JointRefiner,JOINT_DEFAULT
from evaluate_track_stage import CONFIGS
class Fake(JointRefiner):
    def __init__(self,p):self.config={**JOINT_DEFAULT,**CONFIGS['division_local995']};self.p=p
    def predict(self,*args):
        er=np.array([[0,4],[0,5],[1,5],[2,6],[2,7]])
        p=np.array([.999,.999,.01,.01,.999])
        return {'edge':er,'edge_logits':np.log(p/(1-p))[None],'fork':np.array([[0,4,5]]),'fork_logits':np.array([[np.log(self.p/(1-self.p))]])}
nodes={i:{'t':0 if i<4 else 1,'z':0.,'y':0.,'x':float(i)} for i in [0,1,2,4,5,6,7]}
edges=[{'source_id':s,'target_id':d} for s,d in [(0,4),(1,5),(2,6)]]
old={(e['source_id'],e['target_id']) for e in edges}
_,low,_=Fake(.9).refine('synthetic',nodes,edges)
_,high,_=Fake(.99999).refine('synthetic',nodes,edges)
assert {(e['source_id'],e['target_id']) for e in low}==old
assert {(e['source_id'],e['target_id']) for e in high}=={(0,4),(0,5),(2,6)}
print('LOCAL_DIVISION_COUPLING_TEST_PASSED')
