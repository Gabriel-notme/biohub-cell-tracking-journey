from pathlib import Path
import json,torch,numpy as np
from track_video_refine import TrackVideoRefiner
from evaluate_track_stage import CONFIGS
from joint_refine import JOINT_DEFAULT
R=Path('/workspace/biohub');torch.set_num_threads(2)
name=json.loads((R/'track_data/split.json').read_text())['calibration'][0];raw=json.loads((R/'reproduced_B4'/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()};edges=raw['edges']
r=TrackVideoRefiner([R/'Track3_hard_137/best.pt',R/'Track3_hard_823/best.pt'],'/kaggle/input/competitions/biohub-cell-tracking-during-development/train',cache_dir=R/'track_prediction_cache')
r.config={**JOINT_DEFAULT,**CONFIGS['edges_preserve'],'compute_unused_forks':True}
dense=r.predict(name,nodes,edges);_,full,a=r.refine(name,nodes,edges)
r.config.pop('compute_unused_forks');sparse=r.predict(name,nodes,edges);_,fast,b=r.refine(name,nodes,edges)
assert np.array_equal(dense['edge'],sparse['edge']) and np.array_equal(dense['edge_logits'],sparse['edge_logits'])
assert len(sparse['fork'])==0 and len(dense['fork'])>0
canon=lambda ee:sorted((e['source_id'],e['target_id']) for e in ee)
assert canon(full)==canon(fast)
result={'movie':name,'identical_edge_logits':True,'identical_final_edges':True,'eliminated_fork_predictions':len(dense['fork']),'solver_fallback_frames':b['solver_fallback_frames']}
(R/'fork_elision_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
