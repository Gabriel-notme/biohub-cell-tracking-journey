from pathlib import Path
import json,time
R=Path('/workspace/biohub')
base=json.loads((R/'b34_selection.json').read_text())
from evaluate_track_stage import CONFIGS
selected={}
for v,stem in [('B5','Track'),('B6','Track3')]:
    while not all((R/f'{stem}_full_{s}/ready.json').exists() for s in [137,823]):time.sleep(5)
    selection=json.loads(json.dumps(base['B4']))
    selection['stages'].append({'kind':'track_video','model_files':[f'{stem}_full_{s}/best.pt' for s in [137,823]],'config':CONFIGS['joint_strict']})
    selection.update(calibration_frozen=False,purpose='Runtime-only prototype; not selected for competition')
    selected[v]=selection
(R/'prototype_selection_b56.json').write_text(json.dumps(selected,indent=2))
