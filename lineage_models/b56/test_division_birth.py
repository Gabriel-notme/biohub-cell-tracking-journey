import numpy as np,json
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from evaluate_track_stage import CONFIGS
nodes={0:{'t':0,'z':10.,'y':10.,'x':10.},1:{'t':1,'z':10.,'y':10.,'x':9.},2:{'t':1,'z':10.,'y':10.,'x':11.}}
r=JointRefiner.__new__(JointRefiner);p=.99999
r.predict=lambda *args:{'edge':np.array([[0,1],[0,2]]),'fork':np.array([[0,1,2]]),'edge_logits':logit(np.array([[.9,.9]])),'fork_logits':logit(np.array([[p]]))}
r.config={**JOINT_DEFAULT,**CONFIGS['division_direct995']};assert r.refine('x',nodes,[])[1]==[]
r.config={**JOINT_DEFAULT,**CONFIGS['division_birth995']};assert len(r.refine('x',nodes,[])[1])==2
p=.98;assert r.refine('x',nodes,[])[1]==[]
print(json.dumps({'default_unchanged':True,'supported_division_from_endpoint':True,'low_evidence_rejected':True}))
