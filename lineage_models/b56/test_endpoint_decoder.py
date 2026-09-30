import numpy as np,json
from joint_refine import JointRefiner,JOINT_DEFAULT
nodes={0:{'t':0,'z':10.,'y':10.,'x':10.},1:{'t':1,'z':10.,'y':11.,'x':10.},2:{'t':2,'z':10.,'y':12.,'x':10.}}
edges=[{'source_id':1,'target_id':2}]
r=JointRefiner.__new__(JointRefiner)
r.predict=lambda name,nodes,edges:{'edge':np.array([[0,1],[1,2]]),'fork':np.empty((0,3),int),'edge_logits':np.full((1,2),20.),'fork_logits':np.empty((1,0))}
r.config={**JOINT_DEFAULT,'endpoint_recovery':True};_,baseline,_=r.refine('synthetic',nodes,edges);assert len(baseline)==1
r.config.update(endpoint_min_history=1,endpoint_min_future=2,endpoint_motion_error=7,endpoint_max_distance=12,endpoint_probability=.999)
_,changed,_=r.refine('synthetic',nodes,edges);assert {(e['source_id'],e['target_id']) for e in changed}=={(0,1),(1,2)}
print(json.dumps({'legacy_default_preserved':True,'learned_short_track_extension_valid':True}))
