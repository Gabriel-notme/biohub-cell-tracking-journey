"""Reweight supervised fitting examples misclassified by the preceding encoder."""
from pathlib import Path
import numpy as np,json,hashlib,time
from scipy.special import expit
R=Path('/workspace/biohub');O=R/'track_corrupt3';D=R/'track_b4w_3';F=R/'Track3_fusion/features'
while not (O/'ready.json').exists():time.sleep(5)
assert not (R/'TrackRobust_137/training_protocol.json').exists(),'Cannot mutate a running training bank'
split=json.loads((D/'split.json').read_text());stats=[]
for name in split['train']:
    path=F/(name+'.npz')
    while not path.exists():time.sleep(5)
    with np.load(D/(name+'.npz')) as d:bank={k:d[k] for k in d.files}
    with np.load(path) as d:features={k:d[k] for k in d.files}
    row={'movie':name}
    for kind,offset in [('edge',338),('fork',530)]:
        y=bank[kind][:,-1];assert np.array_equal(y,features[kind+'_y'])
        probability=expit(features[kind][:,offset]);error=np.where(y,1-probability,probability)
        bank[kind+'_weight']=(1+19*error**2).astype(np.float32)
        row[kind]={'wrong_confident':int((error>.9).sum()),'mean_importance':float(bank[kind+'_weight'].mean()),'max_importance':float(bank[kind+'_weight'].max())}
    destination=O/(name+'.npz');assert destination.is_symlink(),'Only untouched clean-bank symlinks may be replaced'
    destination.unlink();np.savez_compressed(destination,**bank);stats.append(row)
protocol=json.loads((O/'corruption_protocol.json').read_text());protocol['clean_bank_hard_mining']={'training_only':True,'teacher':'Track3_b4_137/best.pt','teacher_sha256':hashlib.sha256((R/'Track3_b4_137/best.pt').read_bytes()).hexdigest(),'importance':'1 + 19 * supervised_probability_error**2','statistics':stats}
(O/'corruption_protocol.json').write_text(json.dumps(protocol,indent=2));(O/'mining_ready.json').write_text(json.dumps({'movies':len(stats),'wrong_confident':{k:sum(r[k]['wrong_confident'] for r in stats) for k in ['edge','fork']}}));print('ROBUST_SUPERVISED_MINING_DONE',json.dumps(json.loads((O/'mining_ready.json').read_text())),flush=True)
