import numpy as np,json
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
nodes={0:{'t':0,'z':10.,'y':10.,'x':10.},1:{'t':1,'z':10.,'y':11.,'x':10.},2:{'t':1,'z':10.,'y':12.,'x':10.}}
edges=[{'source_id':0,'target_id':1}];r=JointRefiner.__new__(JointRefiner)
prob=np.array([[.4,.999],[.6,.999]],np.float32)
r.predict=lambda *args:{'edge':np.array([[0,1],[0,2]]),'fork':np.empty((0,3),int),'edge_logits':logit(prob),'fork_logits':np.empty((2,0))}
r.config={**JOINT_DEFAULT,'edge_weight':4.,'edge_change_penalty':.1}
_,ee,_=r.refine('x',nodes,edges);assert ee[0]['target_id']==2
r.config.update(edge_consensus_gate=True,edge_old_max=.5,edge_new_min=.995)
_,ee,_=r.refine('x',nodes,edges);assert ee[0]['target_id']==1
prob[:,0]=.1
_,ee,_=r.refine('x',nodes,edges);assert ee[0]['target_id']==2
prob[0,1]=.99
_,ee,_=r.refine('x',nodes,edges);assert ee[0]['target_id']==1
print(json.dumps({'default_behavior_preserved':True,'requires_both_models_reject_old':True,'requires_both_models_accept_new':True}))
