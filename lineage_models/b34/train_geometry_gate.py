"""Training-only, high-recall prefilter for expensive daughter recovery."""
from pathlib import Path
import os,time,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','6')
import numpy as np,lightgbm as lgb,torch
from sklearn.metrics import average_precision_score
from train_events import atomic_save
R=Path('/workspace/biohub');D=R/'predicted_events';O=R/'Geometry_gate';O.mkdir(exist_ok=True)
assert not (O/'best.pt').exists()
while not (D/'ready.json').exists():time.sleep(15)
split=json.loads((D/'split.json').read_text());xs=[];ys=[];groups=[]
for i,n in enumerate(split['train']):
    with np.load(D/(n+'.npz')) as d:
        x=d['fork_geom'];y=d['forks'][:,-1];xs.append(x);ys.append(y);groups.extend([i]*len(y))
x=np.concatenate(xs);y=np.concatenate(ys);groups=np.asarray(groups);oof=np.zeros(len(y));iterations=[]
params={'objective':'binary','metric':'binary_logloss','learning_rate':.05,'num_leaves':15,'max_depth':4,'min_data_in_leaf':25,'lambda_l1':.1,'lambda_l2':15,'scale_pos_weight':20,'feature_fraction':.9,'bagging_fraction':.9,'bagging_freq':1,'num_threads':6,'verbosity':-1,'seed':20261040}
folds=np.asarray([int(hashlib.sha256(n.encode()).hexdigest()[:8],16)%3 for n in split['train']]);reports=[]
for fold in range(3):
    val=folds[groups]==fold;tr=lgb.Dataset(x[~val],label=y[~val]);va=lgb.Dataset(x[val],label=y[val],reference=tr)
    model=lgb.train(params,tr,1200,valid_sets=[va],callbacks=[lgb.early_stopping(100,verbose=False)]);prob=model.predict(x[val]);oof[val]=prob;iterations.append(model.best_iteration)
    report={'fold':fold,'ap':float(average_precision_score(y[val],prob)),'positives':int(y[val].sum()),'iterations':model.best_iteration};reports.append(report);print('GEOMETRY_FOLD',json.dumps(report),flush=True)
threshold=min(.02,float(oof[y>0].min())*.1)
assert threshold>0
model=lgb.train(params,lgb.Dataset(x,label=y),max(100,int(np.median(iterations))))
checkpoint={'config':{'architecture':'geometry_prefilter_lgbm','features':28},'threshold':threshold,'model_string':model.model_to_string(),'training_movies':split['train'],'oof':reports,'oof_positive_retention':float((oof[y>0]>=threshold).mean()),'oof_negative_rejection':float((oof[y==0]<threshold).mean()),'positive':int(y.sum()),'n':len(y),'selection':'threshold is one tenth of the lowest held-out positive probability, capped at 0.02; calibration/audit unused'}
atomic_save(checkpoint,O/'best.pt');(O/'done.json').write_text(json.dumps({k:v for k,v in checkpoint.items() if k!='model_string'},indent=2));print('GEOMETRY_GATE_READY',threshold,checkpoint['oof_negative_rejection'],flush=True)
