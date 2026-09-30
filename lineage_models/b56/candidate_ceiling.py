from pathlib import Path
import json,time
from candidate_rows import candidate_rows
from joint_refine import JOINT_DEFAULT
R=Path('/workspace/biohub');split=json.loads((R/'track_data/split.json').read_text());counts={};start=time.time()
for name in split['calibration']:
    d=json.loads((R/'reproduced_B4'/(name+'.json')).read_text());nodes={int(k):v for k,v in d['nodes'].items()};errors=json.loads((R/'b4_error_analysis'/(name+'_details.json')).read_text())['wrong_links']
    for label,k,dist in [('default',4,14),('wide',8,18),('verywide',12,22)]:
        er,_=candidate_rows(nodes,d['edges'],{**JOINT_DEFAULT,'max_candidates':k,'max_distance':dist},include_forks=False);pairs=set(map(tuple,er.tolist()))
        c=counts.setdefault(label,{'wrong_links':0,'candidate_recoverable':0,'farther_than_radius':0})
        c['wrong_links']+=len(errors);c['candidate_recoverable']+=sum(any((a,b) in pairs for a in e['pred_sources'] for b in e['pred_targets']) for e in errors);c['farther_than_radius']+=sum(e['min_step_um']>dist for e in errors)
    print('CEILING_DONE',name,flush=True)
(R/'candidate_ceiling_calibration.json').write_text(json.dumps({'counts':counts,'seconds':time.time()-start},indent=2));print(json.dumps(counts),flush=True)

