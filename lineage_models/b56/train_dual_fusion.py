"""Train event heads jointly from single-frame and three-frame tracklet embeddings."""
from pathlib import Path
import os,json,time,hashlib
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np,torch,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score
from grouped_validation import grouped_mask
R=Path('/workspace/biohub');O=R/'Track_dualfusion';O.mkdir(exist_ok=True)
assert not (O/'ready.json').exists()
parents=['Track3_fusion','Track_fusion'];encoders=['Track3_b4_137/best.pt','Track_b4_137/best.pt']
while not all((R/p/'ready.json').exists() for p in parents):time.sleep(15)
split=json.loads((R/'track_data/split.json').read_text());models={};metrics={};started=time.time()
for kind in ['edge','fork']:
    sets={};motion_columns=338 if kind=='edge' else 530
    for group in ['train','calibration']:
        xx=[];yy=[];gg=[]
        for mi,name in enumerate(split[group]):
            with np.load(R/parents[0]/'features'/(name+'.npz')) as a,np.load(R/parents[1]/'features'/(name+'.npz')) as b:
                assert np.array_equal(a[kind+'_y'],b[kind+'_y'])
                assert np.array_equal(a[kind][:,:motion_columns],b[kind][:,:motion_columns])
                xx.append(np.c_[a[kind],b[kind][:,motion_columns:]]);yy.append(a[kind+'_y']);gg.extend([mi]*len(a[kind+'_y']))
        sets[group]=(np.concatenate(xx),np.concatenate(yy),np.asarray(gg))
    x,y,groups=sets['train'];vx,vy,_=sets['calibration'];models[kind]=[];metrics[kind]=[]
    for seed in [137,823,2029]:
        params={'objective':'binary','metric':'average_precision','learning_rate':.035,'num_leaves':31,'max_depth':6,'min_data_in_leaf':80,'feature_fraction':.65,'bagging_fraction':.8,'bagging_freq':1,'lambda_l1':1.,'lambda_l2':12.,'num_threads':8,'verbosity':-1,'seed':seed}
        if kind=='fork':params.update(num_leaves=15,min_data_in_leaf=15,lambda_l2=20.,scale_pos_weight=10.)
        mask,fold=grouped_mask(split['train'],groups,y,seed)
        if fold['usable']:
            internal=lgb.train(params,lgb.Dataset(x[~mask],label=y[~mask]),num_boost_round=1800,valid_sets=[lgb.Dataset(x[mask],label=y[mask])],callbacks=[lgb.early_stopping(120,verbose=False)])
            rounds=max(50,internal.best_iteration);internal_ap=float(average_precision_score(y[mask],internal.predict(x[mask])))
        else:rounds=200;internal_ap=None
        booster=lgb.train(params,lgb.Dataset(x,label=y),num_boost_round=rounds)
        prob=booster.predict(vx);metrics[kind].append({'seed':seed,'iterations':rounds,'internal_ap':internal_ap,'fold':fold,'ap':float(average_precision_score(vy,prob)),'auc':float(roc_auc_score(vy,prob))})
        models[kind].append(booster.model_to_string());print('DUAL_FUSION_TRAINED',kind,metrics[kind][-1],flush=True)
protocol={'train':split['train'],'calibration':split['calibration'],'new_audit_excluded':split['new_audit'],'encoders':encoders,'features':'Separate raw single-frame and local-three-frame 3D tracklet embeddings with invariant graph motion; internal movie-grouped round selection, no audit selection.'}
torch.save({'config':{'architecture':'track_video_motion_dual_fusion'},'encoder_sha256_list':[hashlib.sha256((R/n).read_bytes()).hexdigest() for n in encoders],'models':models,'metrics':metrics,'protocol':protocol},O/'best.pt')
(O/'training_protocol.json').write_text(json.dumps(protocol,indent=2));(O/'history.json').write_text(json.dumps(metrics,indent=2));(O/'ready.json').write_text(json.dumps({'seconds':time.time()-started}));print('DUAL_FUSION_DONE',flush=True)
