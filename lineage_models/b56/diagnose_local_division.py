from pathlib import Path
import json,numpy as np
from scipy.special import expit
from refine_events import structure
from track_video_refine import TrackVideoRefiner
from joint_refine import JOINT_DEFAULT
from evaluate_track_stage import CONFIGS,TRAIN
R=Path('/workspace/biohub');name='6bba_2312ac41'
base=json.loads((R/'reproduced_B4'/(name+'.json')).read_text());nodes={int(k):v for k,v in base['nodes'].items()};old,prev,_,pos=structure(nodes,base['edges'])
orig=json.loads((R/'track_evaluation/Track_hold_6bba/joint_strict/graphs'/(name+'.json')).read_text());changed,_,_,_=structure(nodes,orig['edges'])
local=json.loads((R/'track_evaluation/Track_hold_6bba_local_division/division_local995/graphs'/(name+'.json')).read_text());loc,_,_,_=structure(nodes,local['edges'])
obj=TrackVideoRefiner([R/'Track_hold_6bba/best.pt'],TRAIN,cache_dir=R/'track_prediction_cache')
obj.config={**JOINT_DEFAULT,**CONFIGS['division_local995']};pred=obj.predict(name,nodes,base['edges']);fp={tuple(map(int,r)):float(p) for r,p in zip(pred['fork'],expit(pred['fork_logits'].mean(0)))};ep={tuple(map(int,r)):float(p) for r,p in zip(pred['edge'],expit(pred['edge_logits'].mean(0)))}
result=[]
for s,ds in changed.items():
    if len(ds)!=2 or set(ds)==set(old.get(s,[])):continue
    a,b=sorted(ds);owners=[prev.get(d) for d in (a,b)]
    result.append({'old_degree':len(old.get(s,[])),'daughter_owner_degrees':[len(old.get(o,[])) if o is not None and o!=s else 0 for o in owners], 'both_in_local':set(ds)==set(loc.get(s,[])),'fork_probability':fp.get((s,a,b)),'edge_probabilities':[ep.get((s,d)) for d in (a,b)],'distances':[float(np.linalg.norm(pos[d]-pos[s])) for d in ds],'competing_forks':sum(int(r[0])==s for r in pred['fork'])})
print(json.dumps(result,indent=2))
