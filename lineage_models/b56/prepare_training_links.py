from pathlib import Path
import json
R=Path('/workspace/biohub');v=R/'training_videos';v.mkdir(exist_ok=True)
for name in json.loads((R/'track_data/split.json').read_text())['train']:
    p=v/(name+'.zarr')
    if not p.exists():p.symlink_to(Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')/(name+'.zarr'),target_is_directory=True)
print('TRAINING_VIDEO_LINKS_READY')
