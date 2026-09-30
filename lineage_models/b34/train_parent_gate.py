"""Movie-disjoint, high-recall visual parent screening trained without calibration."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,lightgbm as lgb,torch
from sklearn.metrics import average_precision_score
from train_events import atomic_save
R=Path('/workspace/biohub');D=R/'transfer_data';O=R/'Parent_gate';O.mkdir(exist_ok=True)
assert not (O/'best.pt').exists()
split=json.loads((D/'split.json').read_text());xs=[];ys=[];groups=[]
for i,name in enumerate(split['train']):
    with np.load(D/(name+'.npz')) as d:
        sources=np.unique(np.r_[d['edge_rows'][:,0],d['fork_rows'][:,0]])
        positive=np.unique(d['fork_rows'][d['fork_y']>0,0]);y=np.isin(sources,positive).astype(np.int8)
    bank=np.load(D/(name+'_features.npy'),mmap_mode='r');xs.append(np.asarray(bank[sources,1],np.float32));ys.append(y);groups.extend([i]*len(y))
x=np.concatenate(xs);y=np.concatenate(ys);groups=np.asarray(groups);oof=np.zeros(len(y));iterations=[];reports=[]
folds=np.asarray([int(hashlib.sha256(n.encode()).hexdigest()[:8],16)%3 for n in split['train']])
params={'objective':'binary','metric':'average_precision','learning_rate':.035,'num_leaves':15,'max_depth':4,'min_data_in_leaf':30,'lambda_l1':.1,'lambda_l2':10,'scale_pos_weight':5,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'num_threads':4,'verbosity':-1,'seed':20261042}
for fold in range(3):
    val=folds[groups]==fold;tr=lgb.Dataset(x[~val],label=y[~val]);va=lgb.Dataset(x[val],label=y[val],reference=tr)
    model=lgb.train(params,tr,1500,valid_sets=[va],callbacks=[lgb.early_stopping(150,verbose=False)]);prob=model.predict(x[val]);oof[val]=prob;iterations.append(model.best_iteration)
    row={'fold':fold,'ap':float(average_precision_score(y[val],prob)),'positives':int(y[val].sum()),'iterations':model.best_iteration};reports.append(row);print('PARENT_GATE_FOLD',json.dumps(row),flush=True)
thresholds={str(recall):min(.05,float(np.quantile(oof[y>0],1-recall))*.5) for recall in [.98,.995,1.]}
model=lgb.train(params,lgb.Dataset(x,label=y),max(50,int(np.median(iterations))))
report={'config':{'architecture':'parent_visual_prefilter_lgbm','features':64},'thresholds':thresholds,'training_movies':split['train'],'oof':reports,'n':len(y),'positive':int(y.sum()),'retention':{k:{'positive':float((oof[y>0]>=v).mean()),'negative_rejected':float((oof[y==0]<v).mean())} for k,v in thresholds.items()},'selection':'Movie-disjoint OOF thresholds with factor-two safety margin; no calibration or audit labels used.'}
atomic_save({**report,'model_string':model.model_to_string()},O/'best.pt');(O/'done.json').write_text(json.dumps(report,indent=2));print('PARENT_GATE_READY',json.dumps(report['retention']),flush=True)
