from pathlib import Path
import os,json,time,hashlib
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np,torch,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from train_events import atomic_save
R=Path('/workspace/biohub');D=R/'visual_edge_data';O=R/'Visual_edge';O.mkdir(exist_ok=True)
def read(names):
    xs=[];ys=[];groups=[]
    for i,n in enumerate(names):
        with np.load(D/(n+'.npz')) as d:
            if len(d['edge_y']):xs.append(d['edge_x']);ys.append(d['edge_y']);groups.extend([i]*len(d['edge_y']))
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups)
if __name__=='__main__':
    while not (D/'ready.json').exists():time.sleep(15)
    assert not (O/'best.pt').exists();split=json.loads((D/'split.json').read_text());x,y,groups=read(split['train']);xc,yc,_=read(split['calibration']);models=[];reports=[];cp=[]
    # Preserve rotation-invariant motion, supplemented with learned local images.
    x[:,[0,1,2,4,5,6,7,8,9,10,11,12]]=0;xc[:,[0,1,2,4,5,6,7,8,9,10,11,12]]=0
    print('VISUAL_EDGE_DATA',x.shape,int(y.sum()),flush=True)
    for seed in range(3):
        valid=np.asarray([int(hashlib.sha256((n+str(seed)).encode()).hexdigest()[:8],16)%5==0 for n in split['train']]);mask=valid[groups]
        params={'objective':'binary','metric':'binary_logloss','learning_rate':.025,'num_leaves':23,'max_depth':6,'min_data_in_leaf':45,'lambda_l1':.1,'lambda_l2':12,'feature_fraction':.65,'bagging_fraction':.85,'bagging_freq':1,'max_bin':127,'scale_pos_weight':min(15.,np.sqrt((len(y)-y.sum())/max(y.sum(),1))),'num_threads':8,'verbosity':-1,'seed':20261150+seed,'force_col_wise':True}
        tr=lgb.Dataset(x[~mask],label=y[~mask]);va=lgb.Dataset(x[mask],label=y[mask],reference=tr);model=lgb.train(params,tr,2500,valid_sets=[va],callbacks=[lgb.early_stopping(120,verbose=False)]);prob=model.predict(x[mask]);report={'seed':seed,'iterations':model.best_iteration,'ap':float(average_precision_score(y[mask],prob)),'logloss':float(log_loss(y[mask],prob)),'positive':int(y[mask].sum())};reports.append(report);print('VISUAL_EDGE_INTERNAL',json.dumps(report),flush=True)
        final=lgb.train(params,lgb.Dataset(x,label=y),max(100,model.best_iteration));models.append(final.model_to_string());cp.append(final.predict(xc))
    p=np.mean(cp,axis=0);cal={'ap':float(average_precision_score(yc,p)),'auc':float(roc_auc_score(yc,p)),'positive':int(yc.sum()),'n':len(yc)};saved={'config':{'architecture':'visual_motion_edge_lgbm','features':x.shape[1]},'models':models,'internal_validation':reports,'calibration':cal,'training_movies':split['train']};atomic_save(saved,O/'best.pt');(O/'done.json').write_text(json.dumps({k:v for k,v in saved.items() if k!='models'},indent=2));print('VISUAL_EDGE_DONE',json.dumps(cal),flush=True)
