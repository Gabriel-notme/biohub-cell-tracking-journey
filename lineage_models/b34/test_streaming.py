from pathlib import Path
import os,json,time,subprocess,shutil
R=Path('/workspace/biohub');root=R/'stream_equivalence';root.mkdir(exist_ok=True)
data=root/'data';data.mkdir(exist_ok=True);graphs=root/'reference';graphs.mkdir(exist_ok=True)
names=json.loads((R/'events/split.json').read_text())['calibration'][:2]
for name in names:
    link=data/(name+'.zarr')
    if not link.exists():link.symlink_to(R.parent.parent/'kaggle/input/competitions/biohub-cell-tracking-during-development/train'/(name+'.zarr'),target_is_directory=True)
    assert link.exists(),link
env={**os.environ,'BIOHUB_BASE_REPO':str(R/'baseline_validation/tracking_repo')}
cmd=['/workspace/venv/bin/python','-u',str(R/'streaming_pipeline.py'),'--artifact',str(R),'--selection',str(R/'localized_selection.json'),'--variant','B3','--data',str(data),'--graphs',str(graphs),'--work',str(root)]
proc=subprocess.Popen(cmd,env=env,stdout=(root/'stream.log').open('w'),stderr=subprocess.STDOUT)
try:
    for name in names:
        time.sleep(10);tmp=graphs/(name+'.tmp');shutil.copyfile(R/'baseline_graphs'/(name+'.json'),tmp);os.replace(tmp,graphs/(name+'.json'))
    (root/'reference_postprocessing_complete.json').write_text('{}')
    assert proc.wait(timeout=1800)==0
    records=[]
    for name in names:
        source=R/'pipeline_evaluation/localized_motion_cal/graphs'/(name+'.json')
        while not source.exists():time.sleep(5)
        a=json.loads(source.read_text());b=json.loads((root/'lineage_graphs'/(name+'.json')).read_text())
        assert a['nodes']==b['nodes'],name+' coordinates'
        pairs=lambda es:sorted((int(e['source_id']),int(e['target_id'])) for e in es)
        assert pairs(a['edges'])==pairs(b['edges']),name+' edges'
        records.append({'movie':name,'nodes':len(a['nodes']),'edges':len(a['edges']),'exact_graph':True})
    (R/'stream_equivalence.json').write_text(json.dumps({'all_exact':True,'records':records},indent=2));print('STREAM_EQUIVALENCE_PASSED',records,flush=True)
finally:
    if proc.poll() is None:proc.terminate()
