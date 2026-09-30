from pathlib import Path
import json
from evaluate_track_stage import CONFIGS
R=Path('/workspace/biohub');base=json.loads((R/'b34_selection.json').read_text())['B4'];selection={}
for variant in ['B5','B6']:
    candidate=json.loads(json.dumps(base));candidate['stages'].append({'kind':'track_set','model_files':['TrackSetDual_137/best.pt','TrackSetDual_823/best.pt','Track3_b4_137/best.pt','Track_b4_137/best.pt'],'config':CONFIGS['set_consensus']});candidate.update(variant=variant,calibration_frozen=False,runtime_only_unselected=True);selection[variant]=candidate
(R/'runtime_set_selection.json').write_text(json.dumps(selection,indent=2));print('UNSELECTED_RUNTIME_PROTOTYPE_READY')
