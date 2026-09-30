import numpy as np,json
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from evaluate_track_stage import CONFIGS
nodes={0:{'t':0,'z':10.,'y':10.,'x':10.},1:{'t':1,'z':10.,'y':10.,'x':11.}}
edges=[{'source_id':0,'target_id':1}];prob=np.array([[.001],[.002]])
r=JointRefiner.__new__(JointRefiner);r.predict=lambda *args:{'edge':np.array([[0,1]]),'fork':np.empty((0,3),int),'edge_logits':logit(prob),'fork_logits':np.empty((2,0))}
r.config={**JOINT_DEFAULT,**CONFIGS['consensus_wide']};assert len(r.refine('x',nodes,edges)[1])==1
r.config={**JOINT_DEFAULT,**CONFIGS['prune_rejected01']};assert len(r.refine('x',nodes,edges)[1])==0
prob[0,0]=.2;assert len(r.refine('x',nodes,edges)[1])==1
print(json.dumps({'default_preserves':True,'unanimously_rejected_edge_removed':True,'disagreement_preserves':True}))
