from pathlib import Path
import os,json,time,hashlib
os.environ.setdefault('OMP_NUM_THREADS','6')
import numpy as np,torch,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score
from train_events import atomic_save
R=Path('/workspace/biohub');D=R/'visual_motion_data';O=R/'Visual_motion';O.mkdir(exist_ok=True)
def read(names):
    xs=[];ys=[];groups=[]
    for i,n in enumerate(names):
        with np.load(D/(n+'.npz')) as d:
            if len(d['fork_y']):xs.append(d['fork_x']);ys.append(d['fork_y']);groups.extend([i]*len(d['fork_y']))
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups)
if __name__=='__main__':
    while not (D/'ready.json').exists():time.sleep(15)
    assert not (O/'best.pt').exists();split=json.loads((D/'split.json').read_text());x,y,groups=read(split['train']);xc,yc,_=read(split['calibration']);models=[];reports=[];cp=[]
    for seed in range(3):
        valid=np.asarray([int(hashlib.sha256((n+str(seed)).encode()).hexdigest()[:8],16)%5==0 for n in split['train']]);mask=valid[groups]
        params={'objective':'binary','metric':'average_precision','learning_rate':.025,'num_leaves':15,'max_depth':5,'min_data_in_leaf':15,'lambda_l1':.1,'lambda_l2':15,'feature_fraction':.7,'bagging_fraction':.85,'bagging_freq':1,'scale_pos_weight':10,'num_threads':6,'verbosity':-1,'seed':20261050+seed}
        tr=lgb.Dataset(x[~mask],label=y[~mask]);va=lgb.Dataset(x[mask],label=y[mask],reference=tr);model=lgb.train(params,tr,2000,valid_sets=[va],callbacks=[lgb.early_stopping(150,verbose=False)]);prob=model.predict(x[mask]);report={'seed':seed,'iterations':model.best_iteration,'ap':float(average_precision_score(y[mask],prob)),'positive':int(y[mask].sum())};reports.append(report);print('VISUAL_MOTION_INTERNAL',json.dumps(report),flush=True)
        final=lgb.train(params,lgb.Dataset(x,label=y),max(50,model.best_iteration));models.append(final.model_to_string());cp.append(final.predict(xc))
    p=np.mean(cp,axis=0);cal={'ap':float(average_precision_score(yc,p)),'auc':float(roc_auc_score(yc,p)),'positive':int(yc.sum()),'n':len(yc)};saved={'config':{'architecture':'visual_motion_fork_lgbm','features':x.shape[1]},'models':models,'internal_validation':reports,'calibration':cal,'training_movies':split['train']};atomic_save(saved,O/'best.pt');(O/'done.json').write_text(json.dumps({k:v for k,v in saved.items() if k!='models'},indent=2));print('VISUAL_MOTION_DONE',json.dumps(cal),flush=True)
