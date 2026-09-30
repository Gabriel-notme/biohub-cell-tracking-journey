import numpy as np,json
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from evaluate_track_stage import CONFIGS
nodes={0:{'t':0,'z':10.,'y':10.,'x':10.},1:{'t':0,'z':10.,'y':10.,'x':11.},2:{'t':1,'z':10.,'y':10.,'x':11.5},3:{'t':2,'z':10.,'y':10.,'x':12.}}
edges=[{'source_id':1,'target_id':2},{'source_id':2,'target_id':3}]
r=JointRefiner.__new__(JointRefiner);prob=np.array([[.99999,.01,.99],[.99999,.02,.99]])
r.predict=lambda *args:{'edge':np.array([[0,2],[1,2],[2,3]]),'fork':np.empty((0,3),int),'edge_logits':logit(prob),'fork_logits':np.empty((2,0))}
canon=lambda ee:{(e['source_id'],e['target_id']) for e in ee}
r.config={**JOINT_DEFAULT,**CONFIGS['consensus_wide']};_,ee,_=r.refine('x',nodes,edges);assert canon(ee)==canon(edges)
r.config={**JOINT_DEFAULT,**CONFIGS['flow_strict']};_,ee,stats=r.refine('x',nodes,edges);assert canon(ee)=={(0,2),(2,3)} and stats['endpoint_reassignments']==1
prob[0,1]=.8;_,ee,stats=r.refine('x',nodes,edges);assert canon(ee)==canon(edges) and stats['endpoint_reassignments']==0
print(json.dumps({'incumbent_preserved_without_evidence':True,'confident_endpoint_can_replace_rejected_parent':True,'member_disagreement_blocks_reassignment':True}))
