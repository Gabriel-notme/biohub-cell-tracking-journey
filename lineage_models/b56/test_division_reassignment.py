import numpy as np,json
from pathlib import Path
from scipy.special import logit
from joint_refine import JointRefiner,JOINT_DEFAULT
from evaluate_track_stage import CONFIGS
nodes={0:{'t':0,'z':10.,'y':10.,'x':10.},1:{'t':0,'z':10.,'y':10.,'x':15.},2:{'t':1,'z':10.,'y':10.,'x':10.},3:{'t':1,'z':10.,'y':10.,'x':12.}}
edges=[{'source_id':0,'target_id':2},{'source_id':1,'target_id':3}]
r=JointRefiner.__new__(JointRefiner);prob=np.array([[.999,.999,.01],[.999,.999,.02]])
r.predict=lambda *args:{'edge':np.array([[0,2],[0,3],[1,3]]),'fork':np.array([[0,2,3]]),'edge_logits':logit(prob),'fork_logits':logit(np.array([[.99999],[.99999]]))}
canon=lambda ee:{(e['source_id'],e['target_id']) for e in ee}
r.config={**JOINT_DEFAULT,**CONFIGS['division_reassign_strict'],'division_reassignment':False};_,ee,_=r.refine('x',nodes,edges);assert canon(ee)==canon(edges)
r.config['division_reassignment']=True;_,ee,stats=r.refine('x',nodes,edges);assert canon(ee)=={(0,2),(0,3)} and stats['division_target_reassignments']==1
prob[0,2]=.8;_,ee,stats=r.refine('x',nodes,edges);assert canon(ee)==canon(edges) and stats['division_target_reassignments']==0
result={'baseline_unchanged':True,'confident_division_reclaims_wrongly_owned_daughter':True,'disagreement_preserves_incumbent':True}
Path('/workspace/biohub/division_reassignment_tests.json').write_text(json.dumps(result));print(json.dumps(result))
