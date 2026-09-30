import numpy as np
from joint_refine import JointRefiner,JOINT_DEFAULT
from evaluate_track_stage import CONFIGS
class Fake(JointRefiner):
    def __init__(self,threshold):self.config={**JOINT_DEFAULT,**CONFIGS['endpoint_only'+threshold]}
    def predict(self,name,nodes,edges):
        rows=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[6,7],[6,3]])
        p=np.array([.01,.01,.95,.01,.01,.01,.999])
        return {'edge':rows,'edge_logits':np.stack([np.log(p/(1-p))]*2),'fork':np.empty((0,3),int),'fork_logits':np.empty((2,0))}
nodes={i:{'t':i,'z':0.,'y':0.,'x':float(i)} for i in range(6)}
nodes[6]={'t':2,'z':0.,'y':0.,'x':4.};nodes[7]={'t':3,'z':0.,'y':0.,'x':5.}
edges=[{'source_id':s,'target_id':d} for s,d in [(0,1),(1,2),(3,4),(4,5),(6,7)]]
old={(e['source_id'],e['target_id']) for e in edges}
_,strict,_=Fake('98').refine('synthetic',nodes,edges)
_,balanced,_=Fake('90').refine('synthetic',nodes,edges)
assert {(e['source_id'],e['target_id']) for e in strict}==old
assert {(e['source_id'],e['target_id']) for e in balanced}==old|{(2,3)}
print('ENDPOINT_COMPLETION_TEST_PASSED')
