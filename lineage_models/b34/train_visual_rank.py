from pathlib import Path
import os,json,time
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np,torch,lightgbm as lgb
from train_events import atomic_save
R=Path('/workspace/biohub');O=R/'Visual_rank';O.mkdir(exist_ok=True)
def read(names):
    xs=[];ys=[];sizes=[]
    for n in names:
        with np.load(R/'visual_edge_data'/(n+'.npz')) as d:x=d['edge_x'];y=d['edge_y']
        groups=np.load(R/'ranking_groups'/(n+'.npy'))
        for parent in np.unique(groups):
            ix=np.flatnonzero(groups==parent)
            if y[ix].min()==y[ix].max():continue
            xs.append(x[ix]);ys.append(y[ix]);sizes.append(len(ix))
    x=np.concatenate(xs);x[:,[0,1,2,4,5,6,7,8,9,10,11,12]]=0
    return x,np.concatenate(ys).astype(int),np.asarray(sizes)
if __name__=='__main__':
    while not (R/'ranking_groups/ready.json').exists():time.sleep(15)
    assert not (O/'best.pt').exists();split=json.loads((R/'events/split.json').read_text());names=split['train'];domains=sorted({n.split('_')[0] for n in names});parts={d:read([n for n in names if n.startswith(d+'_')]) for d in domains};records=[]
    params={'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[1,2],'label_gain':[0,1],'lambdarank_truncation_level':3,'learning_rate':.025,'num_leaves':15,'max_depth':5,'min_data_in_leaf':50,'lambda_l1':.1,'lambda_l2':15,'feature_fraction':.7,'bagging_fraction':.85,'bagging_freq':1,'max_bin':127,'num_threads':8,'verbosity':-1,'force_col_wise':True}
    for domain in domains:
        tx,ty,tg=parts[[d for d in domains if d!=domain][0]];vx,vy,vg=parts[domain];tr=lgb.Dataset(tx,label=ty,group=tg);va=lgb.Dataset(vx,label=vy,group=vg,reference=tr)
        m=lgb.train({**params,'seed':20261300},tr,1800,valid_sets=[va],callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False)]);record={'held_out_embryo':domain,'iterations':m.best_iteration,'metrics':m.best_score['valid_0']};records.append(record);print('RANK_DOMAIN_VALIDATION',json.dumps(record),flush=True)
    iterations=max(100,int(np.median([r['iterations'] for r in records])));x=np.concatenate([parts[d][0] for d in domains]);y=np.concatenate([parts[d][1] for d in domains]);group=np.concatenate([parts[d][2] for d in domains]);models=[]
    for seed in range(3):
        m=lgb.train({**params,'seed':20261310+seed,'bagging_seed':1310+seed,'feature_fraction_seed':3130+seed},lgb.Dataset(x,label=y,group=group),iterations);models.append(m.model_to_string())
    result={'config':{'architecture':'visual_edge_ranker','features':x.shape[1],'iterations':iterations},'models':models,'training_movies':names,'domain_validation':records};atomic_save(result,O/'best.pt');(O/'done.json').write_text(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2));print('VISUAL_RANK_DONE',iterations,flush=True)
