from pathlib import Path
import os,json,subprocess,time
from runtime_batch_patch import make_baseline_batched
R=Path('/workspace/biohub');C=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development');PY='/workspace/venv/bin/python'
while not (R/'data_ready.json').exists():time.sleep(5)
split=json.loads((R/'events/split.json').read_text())
import hashlib
fresh=[]
for embryo in ['44b6','6bba']:
    fresh.extend(sorted([n for n in split['train'] if n.startswith(embryo)],key=lambda n:hashlib.sha256(('B56-audit-20260923:'+n).encode()).hexdigest())[:16])
for label,names in [('validation',split['calibration']+split['audit']),('newaudit',fresh),('training',[n for n in split['train'] if n not in fresh])]:
    v=R/(label+'_videos');v.mkdir(exist_ok=True)
    for name in names:
        p=v/(name+'.zarr')
        if not p.exists():p.symlink_to(C/'train'/(name+'.zarr'),target_is_directory=True)
    s=(R/'cloud_baseline.py').read_text()
    s=s.replace("if input_root.exists():\n        candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))","if input_root.exists() and not any(p.is_file() for p in candidates):\n        candidates.extend(input_root.rglob('ARTIFACT_MANIFEST.json'))")
    s=s.replace("if input_root.exists():\n        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):","if input_root.exists() and not any(p.is_file() for p in candidates):\n        for name in ('checkpoint_last.pt', 'best.pt', 'last.pt'):")
    os.environ['BIOHUB_D4_BATCH_SIZE']='4';s=make_baseline_batched(s)
    script=R/(label+'_reference.py');script.write_text(s)
    env={**os.environ,'BIOHUB_DATA_ROOT':str(v),'BIOHUB_WORKDIR':str(R/('baseline_'+label)),
        'BIOHUB_GRAPH_CACHE':str(R/('baseline_graphs' if label=='validation' else 'training_baseline_graphs')),
        'BIOHUB_UNET_BATCH_SIZE':'8','BIOHUB_ALLOW_PIP_INSTALL':'1','BIOHUB_VALIDATOR_ENABLE':'0','BIOHUB_EVENT_ENABLED':'0','POLARS_MAX_THREADS':'3','OMP_NUM_THREADS':'3','MKL_NUM_THREADS':'3'}
    subprocess.run([PY,'-u',str(script)],env=env,cwd=R,check=True)
    (R/(label+'_reference_ready.json')).write_text(json.dumps({'movies':len(names),'time':time.time()}))
    print('REFERENCE_READY',label,flush=True)
