from pathlib import Path
import os,json,argparse
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,torch,lightgbm as lgb
R=Path('/workspace/biohub');p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();O=R/a.model
ckpt=torch.load(O/'best.pt',map_location='cpu',weights_only=False);names=ckpt['protocol']['calibration'];result={}
for kind in ['edge','fork']:
    models=[lgb.Booster(model_str=s) for s in ckpt['models'][kind]];ys=[];prob=[]
    for name in names:
        with np.load(O/'features'/(name+'.npz')) as d:
            x=d[kind]
            if ckpt['config'].get('drop_incumbent'):
                from track_fusion import drop_incumbent_features
                x=drop_incumbent_features(x,kind)
            ys.append(d[kind+'_y']);prob.append(np.mean([m.predict(x,num_threads=2) for m in models],0))
    y=np.concatenate(ys);q=np.concatenate(prob);order=np.argsort(-q)
    result[kind]={'positives':int(y.sum()),'positive_quantiles':np.quantile(q[y==1],[0,.1,.5,.9,1]).tolist(),'negative_quantiles':np.quantile(q[y==0],[.99,.999,.9999,1]).tolist(),'threshold_counts':{str(t):{'tp':int(((q>=t)&(y==1)).sum()),'fp':int(((q>=t)&(y==0)).sum())} for t in [.05,.1,.2,.5,.8,.95,.995]},'topk_precision':{str(k):float(y[order[:k]].mean()) for k in [5,10,20,50]}}
(O/'probability_audit.json').write_text(json.dumps(result,indent=2));print(a.model,json.dumps(result))
