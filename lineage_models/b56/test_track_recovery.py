from pathlib import Path
import json,torch,numpy as np
from track_recovery import TrackRecoveryRefiner
from recover_joint import RecoveryRefiner
from refine_events import structure
from cell_event import SCALE,chain
R=Path('/workspace/biohub');names=json.loads((R/'track_data/split.json').read_text())['calibration'];sel=json.loads((R/'b34_selection.json').read_text())['B4'];proposal=[s for s in sel['stages'] if s['kind']=='verified_recovery'][0]['model_files'][0]
model=TrackRecoveryRefiner([R/proposal,R/'Track3_full_137/best.pt'],Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train'),{'heat_threshold':0.,'edge_threshold':0.,'fork_threshold':0.,'proposal_min_probability':0.,'arrival_threshold':0.},R/'recovery_test_cache');tested={};original_method=RecoveryRefiner.proposals
for name in names:
    raw=json.loads((R/'reproduced_B4'/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()};edges=raw['edges'];out,prev,frames,pos=structure(nodes,edges)
    for kind,degree in [('edge',1),('fork',2)]:
        if kind in tested:continue
        choices=[(s,q,out[q][0]) for s,ds in out.items() if len(ds)==degree and len(chain(s,prev,pos))>=3 for q in ds if len(out.get(q,[]))==1 and len(chain(out[q][0],out,pos))>=3]
        if not choices:continue
        s,q,d=choices[0];nn={n:v for n,v in nodes.items() if n!=q};ee=[e for e in edges if q not in [e['source_id'],e['target_id']]]
        record={'s':s,'d':d,'position':pos[q].tolist(),'heat':1.,'kind':kind,'probability':1.,'old':[n for n in out[s] if n!=q]}
        RecoveryRefiner.proposals=lambda self,name,nodes,edges:[record]
        try:n2,e2,stats=model.refine(name,nn,ee)
        finally:RecoveryRefiner.proposals=original_method
        assert len(n2)==len(nodes) and len(e2)==len(edges),(kind,stats)
        added=set(n2)-set(nn);assert len(added)==1
        restored=n2[next(iter(added))];assert np.allclose([restored[k] for k in ['z','y','x']],[nodes[q][k] for k in ['z','y','x']],atol=1e-5)
        tested[kind]={'movie':name,'restored_one_node':True,'restored_original_edge_count':True,'seconds':stats['seconds']}
    if len(tested)==2:break
assert len(tested)==2
(R/'track_recovery_tests.json').write_text(json.dumps(tested,indent=2));print(json.dumps(tested))
