"""Prioritize verified B4 mistakes using only the fitting split's annotations."""
from pathlib import Path
import numpy as np,json,time
R=Path('/workspace/biohub');D=R/'track_b4w_3';O=R/'track_residual3';O.mkdir(exist_ok=True)
while not (D/'ready.json').exists():time.sleep(10)
split=json.loads((D/'split.json').read_text());(O/'split.json').write_text(json.dumps(split));stats=[]
for group in ['train','calibration']:
    for name in split[group]:
        destination=O/(name+'.npz')
        if destination.exists():continue
        if group=='calibration':destination.symlink_to(D/(name+'.npz'))
        else:
            with np.load(D/(name+'.npz')) as d:bank={k:d[k] for k in d.files}
            raw=json.loads((R/'b4_training_graphs'/(name+'.json')).read_text());old={(int(e['source_id']),int(e['target_id'])) for e in raw['edges']};out={}
            for s,d in old:out.setdefault(s,set()).add(d)
            rowstat={'movie':name}
            for kind in ['edge','fork']:
                rr=bank['ids'][bank[kind][:,:-1]];y=bank[kind][:,-1]
                inherited=np.asarray([(tuple(r) in old) if kind=='edge' else set(r[1:])==out.get(int(r[0]),set()) for r in rr])
                weight=np.ones(len(rr),np.float32);bad_old=inherited&(y==0);missed=(~inherited)&(y==1)
                weight[bad_old]=20 if kind=='edge' else 50;weight[missed]=5
                bank[kind+'_weight']=weight;rowstat[kind]={'wrong_inherited':int(bad_old.sum()),'missed_truth':int(missed.sum()),'examples':len(rr)}
            np.savez_compressed(destination,**bank);stats.append(rowstat)
        crops=O/(name+'_crops.npy')
        if not crops.exists():crops.symlink_to(D/(name+'_crops.npy'))
(O/'importance_protocol.json').write_text(json.dumps({'training_only':True,'wrong_existing_edge_weight':20,'wrong_existing_fork_weight':50,'missed_positive_weight':5,'movies':stats},indent=2))
(O/'ready.json').write_text(json.dumps({'movies':len(split['train'])+len(split['calibration'])}));print('RESIDUAL_BANK_READY',flush=True)
