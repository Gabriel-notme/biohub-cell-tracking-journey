from pathlib import Path
import json
from evaluate_track_stage import CONFIGS
R=Path('/workspace/biohub');base=json.loads((R/'b34_selection.json').read_text())['B4'];selection={}
for variant in ['B5','B6']:
    candidate=json.loads(json.dumps(base));candidate['stages'].append({'kind':'track_rank','model_files':['TrackRankDual/best.pt','Track3_b4_137/best.pt','Track_b4_137/best.pt'],'config':CONFIGS['set_consensus']});candidate.update(variant=variant,calibration_frozen=False,runtime_only_unselected=True);selection[variant]=candidate
(R/'runtime_rank_selection.json').write_text(json.dumps(selection,indent=2));print('UNSELECTED_RANKING_RUNTIME_READY')
