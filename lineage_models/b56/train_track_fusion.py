from pathlib import Path
import os,json,time,argparse,hashlib
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,torch,lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score
from track_fusion import encode_bank,fusion_features,drop_incumbent_features
from track_video import load_track
from fast_motion_features import FastMotionFeatures
from grouped_validation import grouped_mask

R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--residual-importance',action='store_true');p.add_argument('--drop-incumbent',action='store_true');p.add_argument('--feature-parent');p.add_argument('--hold-embryo',default='');p.add_argument('--encoder',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    D=R/a.data;O=R/a.out;O.mkdir(exist_ok=True);torch.set_num_threads(4)
    while not (D/'ready.json').exists() or not (R/a.encoder).parent.joinpath('ready.json').exists():time.sleep(10)
    assert not (O/'ready.json').exists();model,ckpt=load_track(R/a.encoder)
    encoder_sha=hashlib.sha256((R/a.encoder).read_bytes()).hexdigest();split=json.loads((D/'split.json').read_text())
    if a.hold_embryo:
        split['train']=[n for n in split['train'] if not n.startswith(a.hold_embryo)]
        split['calibration']=[n for n in split['calibration'] if n.startswith(a.hold_embryo)]
    cache=O/'features'
    if a.feature_parent:
        parent=R/a.feature_parent;prior=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)
        assert prior['encoder_sha256']==encoder_sha and prior['protocol']['train']==split['train'] and prior['protocol']['calibration']==split['calibration']
        if not cache.exists():cache.symlink_to(parent/'features',target_is_directory=True)
    cache.mkdir(exist_ok=True);started=time.time()
    for group in ['train','calibration']:
        for name in split[group]:
            dest=cache/(name+'.npz')
            if dest.exists():continue
            with np.load(D/(name+'.npz')) as d:sample={k:d[k] for k in d.files}
            crops=np.load(D/(name+'_crops.npy'),mmap_mode='r');bank=encode_bank(model,crops,sample['seq'],sample['motion'])
            source=R/('b4_training_graphs' if group=='train' else 'reproduced_B4')/(name+'.json');raw=json.loads(source.read_text())
            nodes={int(k):v for k,v in raw['nodes'].items()};mf=FastMotionFeatures(name,nodes,raw['edges'],TRAIN,R/'track_fusion_motion_cache')
            result={}
            for kind in ['edge','fork']:
                rows=sample[kind];rr=sample['ids'][rows[:,:-1]];m=mf.rows(kind,rr)
                result[kind]=fusion_features(model,bank,rows[:,:-1],sample[kind+'_geom'],m,kind);result[kind+'_y']=rows[:,-1]
            np.savez_compressed(dest,**result);print('FUSION_FEATURES',name,round(time.time()-started),flush=True)
    models={};metrics={}
    for kind in ['edge','fork']:
        sets={}
        for group in ['train','calibration']:
            xx=[];yy=[];gg=[]
            for movie_index,name in enumerate(split[group]):
                with np.load(cache/(name+'.npz')) as d:xx.append(d[kind]);yy.append(d[kind+'_y']);gg.extend([movie_index]*len(d[kind+'_y']))
            sets[group]=(np.concatenate(xx),np.concatenate(yy),np.asarray(gg))
        x,y,groups=sets['train'];vx,vy,_=sets['calibration'];models[kind]=[];metrics[kind]=[]
        importance=np.ones(len(y),np.float32)
        if a.residual_importance:
            weights=[];labels=[]
            for name in split['train']:
                with np.load(R/'track_residual3'/(name+'.npz')) as d:weights.append(d[kind+'_weight']);labels.append(d[kind][:,-1])
            assert np.array_equal(np.concatenate(labels),y),'Residual rows must match cached features'
            importance=np.concatenate(weights)
        if a.drop_incumbent:x=drop_incumbent_features(x,kind);vx=drop_incumbent_features(vx,kind)
        for seed in [137,823,2029]:
            params={'objective':'binary','metric':'average_precision','learning_rate':.035,'num_leaves':31,'max_depth':6,'min_data_in_leaf':80,'feature_fraction':.75,'bagging_fraction':.8,'bagging_freq':1,'lambda_l1':1.,'lambda_l2':8.,'num_threads':8,'verbosity':-1,'seed':seed,'scale_pos_weight':min(30.,(len(y)-y.sum())/max(1,y.sum())) if kind=='fork' else 1.}
            if kind=='fork':params.update(num_leaves=15,min_data_in_leaf=15,lambda_l2=15.,scale_pos_weight=10.)
            mask,fold=grouped_mask(split['train'],groups,y,seed)
            if fold['usable']:
                internal=lgb.train(params,lgb.Dataset(x[~mask],label=y[~mask],weight=importance[~mask]),num_boost_round=1800,valid_sets=[lgb.Dataset(x[mask],label=y[mask])],callbacks=[lgb.early_stopping(120,verbose=False)])
                rounds=max(50,internal.best_iteration);internal_ap=float(average_precision_score(y[mask],internal.predict(x[mask])))
            else:rounds=200;internal_ap=None
            booster=lgb.train(params,lgb.Dataset(x,label=y,weight=importance),num_boost_round=rounds)
            prob=booster.predict(vx);metrics[kind].append({'seed':seed,'iterations':rounds,'internal_ap':internal_ap,'fold':fold,'ap':float(average_precision_score(vy,prob)),'auc':float(roc_auc_score(vy,prob))})
            models[kind].append(booster.model_to_string());print('FUSION_TRAINED',kind,metrics[kind][-1],flush=True)
    protocol={'encoder':a.encoder,'encoder_sha256':encoder_sha,'train':split['train'],'calibration':split['calibration'],'new_audit_excluded':split['new_audit'],'model_features':'Trained ten-frame 3D image-track transformer embeddings and invariant motion descriptors; gradient boosted event heads. Encoder and heads fit training only; encoder checkpoint uses calibration, tree rounds use movie-grouped internal training validation.'}
    protocol.update(residual_importance=a.residual_importance,drop_incumbent=a.drop_incumbent,feature_parent=a.feature_parent)
    torch.save({'config':{'architecture':'track_video_motion_fusion','drop_incumbent':a.drop_incumbent},'encoder_sha256':encoder_sha,'models':models,'metrics':metrics,'protocol':protocol},O/'best.pt')
    (O/'training_protocol.json').write_text(json.dumps(protocol,indent=2));(O/'history.json').write_text(json.dumps(metrics,indent=2));(O/'ready.json').write_text(json.dumps({'seconds':time.time()-started}));print('TRACK_FUSION_DONE',flush=True)
