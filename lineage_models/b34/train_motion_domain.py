from pathlib import Path
import os,json,time
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np,torch,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from train_motion import read
from train_events import atomic_save
from motion_features import invariant_features,VERSION
R=Path('/workspace/biohub');O=R/'Motion_domain';O.mkdir(exist_ok=True)
assert not (O/'best.pt').exists()
split=json.loads((R/'events/split.json').read_text());names=split['train'];x,y,groups=read(names,'edge');x=invariant_features(x,'edge');xc,yc,_=read(split['calibration'],'edge');xc=invariant_features(xc,'edge');domains=sorted({n.split('_')[0] for n in names});assert len(domains)==2
params={'objective':'binary','metric':'binary_logloss','learning_rate':.025,'num_leaves':15,'max_depth':5,'min_data_in_leaf':60,'lambda_l1':.2,'lambda_l2':15,'feature_fraction':.75,'bagging_fraction':.85,'bagging_freq':1,'max_bin':127,'num_threads':8,'verbosity':-1,'force_col_wise':True,'scale_pos_weight':min(15.,np.sqrt((len(y)-y.sum())/max(1,y.sum())))}
records=[]
for domain in domains:
    mask=np.asarray([names[g].startswith(domain+'_') for g in groups]);tr=lgb.Dataset(x[~mask],label=y[~mask]);va=lgb.Dataset(x[mask],label=y[mask],reference=tr)
    m=lgb.train({**params,'seed':20260923},tr,2000,valid_sets=[va],callbacks=[lgb.early_stopping(120,verbose=False)]);p=m.predict(x[mask]);record={'held_out_embryo':domain,'iterations':m.best_iteration,'ap':float(average_precision_score(y[mask],p)),'auc':float(roc_auc_score(y[mask],p)),'logloss':float(log_loss(y[mask],p))};records.append(record);print('CROSS_EMBRYO_VALIDATION',json.dumps(record),flush=True)
iterations=max(100,int(np.median([r['iterations'] for r in records])));saved=torch.load(R/'Motion_invariant/best.pt',map_location='cpu',weights_only=False);saved['models']['edge']=[];cal=[]
for seed in range(3):
    m=lgb.train({**params,'seed':20261200+seed,'bagging_seed':1200+seed,'feature_fraction_seed':2100+seed},lgb.Dataset(x,label=y),iterations);saved['models']['edge'].append(m.model_to_string());cal.append(m.predict(xc))
prob=np.mean(cal,0);saved['config'].update(architecture='graph_motion_lightgbm',rotation_invariant=True,feature_version=VERSION,validation_strategy='leave_one_training_embryo_out');saved['domain_validation']=records;saved['internal_validation']['edge']=records;saved['calibration']['edge']={'ap':float(average_precision_score(yc,prob)),'auc':float(roc_auc_score(yc,prob)),'logloss':float(log_loss(yc,prob)),'n':len(yc)}
atomic_save(saved,O/'best.pt');(O/'done.json').write_text(json.dumps({'iterations':iterations,'domain_validation':records,'calibration':saved['calibration']['edge']},indent=2));print('DOMAIN_MOTION_DONE',json.dumps(saved['calibration']['edge']),flush=True)
