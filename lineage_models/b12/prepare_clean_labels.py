from pathlib import Path
import json,numpy as np,shutil,time
from prepare_events import graph
root=Path('/workspace/biohub');source=root/'events_temporal11';dest=root/'events_clean11';dest.mkdir(exist_ok=True)
split=json.loads((source/'split.json').read_text());shutil.copy2(source/'split.json',dest/'split.json');reports=[]
for name in split['train']+split['calibration']+split['audit']:
    bank=dest/(name+'_patches.npy')
    if not bank.exists():bank.symlink_to(source/(name+'_patches.npy'))
    output=dest/(name+'.npz')
    if name in split['audit']:
        if not output.exists():output.symlink_to(source/(name+'.npz'))
        continue
    nodes,out,prev,frames,pos=graph(Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')/(name+'.geff'))
    with np.load(source/(name+'.npz')) as z:data={k:z[k] for k in z.files}
    ids=data['node_ids'];edges=data['edges'];detpos={}
    for row,geom in zip(edges,data['edge_geom']):
        s,d,label=map(int,row)
        if int(ids[s]) in pos and int(ids[d]) not in pos:detpos[d]=pos[int(ids[s])]+geom[:3]*10
    def uncertain(row):
        s,*children,label=map(int,row)
        if label:return False
        truth=out.get(int(ids[s]),[])
        if not truth:return False
        return any(d in detpos and min(np.linalg.norm(detpos[d]-pos[c]) for c in truth)<=7 for d in children)
    removed={}
    for key,gkey in [('edges','edge_geom'),('forks','fork_geom')]:
        keep=np.array([not uncertain(row) for row in data[key]])
        removed[key]=int((~keep).sum());data[key]=data[key][keep];data[gkey]=data[gkey][keep]
    np.savez_compressed(output,**data)
    reports.append({'movie':name,'split':'train' if name in split['train'] else 'calibration','ignored_ambiguous_edges':removed['edges'],'ignored_ambiguous_forks':removed['forks']})
(dest/'label_quality_report.json').write_text(json.dumps({'criterion':'Ignore detector-mined negative events within 7 physical micrometres of any annotated true daughter; do not relabel as positives. Known annotated identities are retained.','movies':reports},indent=2))
(dest/'ready.json').write_text(json.dumps({'time':time.time()}));print('CLEAN_LABELS_READY',len(reports),flush=True)
