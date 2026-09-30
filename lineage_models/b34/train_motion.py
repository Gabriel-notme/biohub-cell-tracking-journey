from pathlib import Path
import os,json,time,argparse,hashlib
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np,torch,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from train_events import atomic_save
from motion_features import VERSION,invariant_features
R=Path('/workspace/biohub');DATA=R/'motion_data'

def read(names,kind):
    xs=[];ys=[];groups=[]
    for i,name in enumerate(names):
        with np.load(DATA/(name+'.npz')) as d:
            if len(d[kind+'_y']):xs.append(d[kind+'_x']);ys.append(d[kind+'_y']);groups.extend([i]*len(d[kind+'_y']))
    return np.concatenate(xs),np.concatenate(ys),np.asarray(groups)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/workspace/biohub/Motion_v1');p.add_argument('--seeds',type=int,default=3);p.add_argument('--rotation-invariant',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True)
    assert not (out/'best.pt').exists(),'Refuse to overwrite model'
    while not (DATA/'ready.json').exists():time.sleep(20)
    split=json.loads((DATA/'split.json').read_text());names=split['train'];assert not set(names)&set(split['calibration']+split['audit']);saved={'config':{'architecture':'graph_motion_lightgbm','feature_version':VERSION,'lightgbm_version':lgb.__version__},'models':{},'training_movies':names,'internal_validation':{},'calibration':{}}
    for kind in ['edge','fork']:
        x,y,group=read(names,kind);xc,yc,_=read(split['calibration'],kind);positive=int(y.sum());print('MOTION_FIT_DATA',kind,x.shape,positive,flush=True)
        saved['config']['rotation_invariant']=a.rotation_invariant
        if a.rotation_invariant:x=invariant_features(x,kind);xc=invariant_features(xc,kind)
        params={'objective':'binary','metric':['binary_logloss','auc'],'learning_rate':.025,'num_leaves':23 if kind=='edge' else 15,'max_depth':6 if kind=='edge' else 5,'min_data_in_leaf':45 if kind=='edge' else 12,'lambda_l1':.1,'lambda_l2':8.,'feature_fraction':.8,'bagging_fraction':.85,'bagging_freq':1,'max_bin':127,'num_threads':8,'verbosity':-1,'force_col_wise':True,'scale_pos_weight':min(15.,np.sqrt((len(y)-positive)/max(positive,1)))}
        saved['models'][kind]=[];inner=[];predictions=[]
        for seed in range(a.seeds):
            valid=np.asarray([int(hashlib.sha256((n+str(seed)).encode()).hexdigest()[:8],16)%5==0 for n in names]);mask=valid[group]
            tr=lgb.Dataset(x[~mask],label=y[~mask]);va=lgb.Dataset(x[mask],label=y[mask],reference=tr);pp={**params,'seed':20260923+seed,'bagging_seed':923+seed,'feature_fraction_seed':329+seed}
            model=lgb.train(pp,tr,2500,valid_sets=[va],callbacks=[lgb.early_stopping(120,first_metric_only=True,verbose=False),lgb.log_evaluation(300)])
            pred=model.predict(x[mask]);record={'seed':seed,'iterations':model.best_iteration,'positive':int(y[mask].sum()),'ap':float(average_precision_score(y[mask],pred)),'auc':float(roc_auc_score(y[mask],pred)),'logloss':float(log_loss(y[mask],pred))};inner.append(record);print('MOTION_INTERNAL_VALIDATION',kind,json.dumps(record),flush=True)
            final=lgb.train(pp,lgb.Dataset(x,label=y),max(100,model.best_iteration));saved['models'][kind].append(final.model_to_string());predictions.append(final.predict(xc))
        prob=np.mean(predictions,axis=0);saved['internal_validation'][kind]=inner;saved['calibration'][kind]={'ap':float(average_precision_score(yc,prob)),'auc':float(roc_auc_score(yc,prob)),'logloss':float(log_loss(yc,prob)),'positive':int(yc.sum()),'n':len(yc)}
        print('MOTION_CALIBRATION_CLASSIFICATION',kind,json.dumps(saved['calibration'][kind]),flush=True)
        atomic_save(saved,out/'partial.pt')
    atomic_save(saved,out/'best.pt');(out/'done.json').write_text(json.dumps({'finished':time.time(),'training_movies':len(names),'internal_validation':saved['internal_validation'],'calibration':saved['calibration']},indent=2));print('MOTION_TRAINING_COMPLETE',flush=True)
