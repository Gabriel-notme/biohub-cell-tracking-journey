"""Learn source-conditioned ordering directly with movie-held-out ranking loss."""
from pathlib import Path
import os,json,time,hashlib
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np,torch,lightgbm as lgb
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from grouped_validation import grouped_mask
R=Path('/workspace/biohub');O=R/'TrackRankDual';O.mkdir(exist_ok=True);assert not (O/'ready.json').exists();split=json.loads((R/'track_data/split.json').read_text());encoders=['Track3_b4_137/best.pt','Track_b4_137/best.pt'];xx=[];yy=[];sizes=[];query_movie=[];ww=[]
for mi,name in enumerate(split['train']):
    with np.load(R/'Track3_fusion/features'/(name+'.npz')) as a,np.load(R/'Track_fusion/features'/(name+'.npz')) as b:x=np.c_[a['edge'],b['edge'][:,338:]];y=a['edge_y'];assert np.array_equal(y,b['edge_y'])
    with np.load(R/'track_b4w_3'/(name+'.npz')) as d:source=d['edge'][:,0]
    for parent in np.unique(source):
        ix=np.flatnonzero(source==parent)
        if y[ix].sum()==0 or y[ix].sum()==len(ix):continue
        old=x[ix,45]>.5;error=bool(np.any(old&(y[ix]==0)) or np.any((~old)&(y[ix]>0)));block=x[ix].copy();block[:,44]=.5;block[:,45]=0.
        xx.append(block);yy.append(y[ix]);ww.append(np.full(len(ix),12. if error else 1.,np.float32));sizes.append(len(ix));query_movie.append(mi)
x=np.concatenate(xx);y=np.concatenate(yy).astype(int);weights=np.concatenate(ww);sizes=np.asarray(sizes);query_movie=np.asarray(query_movie);row_movie=np.repeat(query_movie,sizes);models=[];temperatures=[];reports=[];started=time.time()
def temperature(scores,labels,groups):
    starts=np.r_[0,np.cumsum(groups)];queries=[(scores[s:e],labels[s:e]/labels[s:e].sum()) for s,e in zip(starts[:-1],starts[1:])]
    def objective(t):return float(np.mean([logsumexp(s/t)-np.dot(target,s/t) for s,target in queries]))
    optimum=minimize_scalar(objective,bounds=(.25,8.),method='bounded');return float(optimum.x),float(optimum.fun)
for seed in [137,823,2029]:
    held,fold=grouped_mask(split['train'],row_movie,y,seed);assert fold['usable'];query_held=np.isin(query_movie,np.unique(row_movie[held]));train_groups=sizes[~query_held];valid_groups=sizes[query_held]
    params={'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[1,2],'label_gain':[0,1],'lambdarank_truncation_level':3,'learning_rate':.035,'num_leaves':31,'max_depth':6,'min_data_in_leaf':60,'lambda_l1':1.,'lambda_l2':12.,'feature_fraction':.75,'num_threads':8,'verbosity':-1,'seed':seed,'force_col_wise':True}
    internal=lgb.train(params,lgb.Dataset(x[~held],label=y[~held],group=train_groups,weight=weights[~held]),num_boost_round=1600,valid_sets=[lgb.Dataset(x[held],label=y[held],group=valid_groups)],callbacks=[lgb.early_stopping(100,verbose=False)])
    rounds=max(50,internal.best_iteration);temp,nll=temperature(internal.predict(x[held]),y[held],valid_groups)
    final=lgb.train(params,lgb.Dataset(x,label=y,group=sizes,weight=weights),num_boost_round=rounds);models.append(final.model_to_string());temperatures.append(temp);reports.append({'seed':seed,'iterations':rounds,'temperature':temp,'heldout_query_nll':nll,'fold':fold});print('RANK_TRAINED',seed,rounds,temp,nll,flush=True)
protocol={'train':split['train'],'calibration':split['calibration'],'new_audit_excluded':split['new_audit'],'internal_folds':reports,'objective':'LambdaRank over each parent and all candidate daughters, using complementary trained image tracklets and motion; mistaken inherited assignments weighted 12x. Entire movies select rounds and temperature; no calibration or audit labels used in this training. Inherited probability and membership features masked.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
torch.save({'config':{'architecture':'candidate_rank','input_dim':x.shape[1]},'models':models,'temperatures':temperatures,'protocol':protocol,'encoder_sha256_list':[hashlib.sha256((R/p).read_bytes()).hexdigest() for p in encoders]},O/'best.pt');(O/'training_protocol.json').write_text(json.dumps(protocol,indent=2));(O/'history.json').write_text(json.dumps(reports,indent=2));(O/'ready.json').write_text(json.dumps({'seconds':time.time()-started,'queries':len(sizes),'rows':len(y)}));print('RANK_TRAINING_DONE',flush=True)
