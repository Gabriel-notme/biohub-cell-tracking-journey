"""Cache frozen encoder outputs on reserved images, without reading annotations or scores.

This performs no fitting, candidate-head inference, selection, or metric evaluation.
It only avoids repeating shared encoding after the two configurations are frozen.
"""
from pathlib import Path
import os,json,hashlib,argparse,time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,torch
from track_video import load_track,TrackMovie,sequences
from track_fusion import encode_bank,fusion_features
from track_video_refine import geometries
from candidate_rows import candidate_rows
from joint_refine import JOINT_DEFAULT
from refine_events import structure
from fast_motion_features import FastMotionFeatures
from motion_features import invariant_features
R=Path('/workspace/biohub');TRAIN=Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/train');C=R/'track_prediction_cache'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='B4',choices=['B3','B4','B3visual','Teachers']);p.add_argument('--shard',type=int,required=True);p.add_argument('--encoder-group',default='dual',choices=['dual','robust']);p.add_argument('--group',default='new_audit',choices=['new_audit','audit','calibration']);a=p.parse_args();torch.set_num_threads(2)
    names=json.loads((R/'track_data/split.json').read_text())[a.group];paths=[R/'Track3_b4_137/best.pt',R/'Track_b4_137/best.pt'];models=[]
    if a.encoder_group=='robust':paths=[R/'TrackRobust_137/best.pt']
    for path in paths:
        model,ck=load_track(path);assert not set(ck['protocol']['train'])&set(names);models.append(model)
    signatures=[hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in paths]
    config={**JOINT_DEFAULT,'max_distance':18.,'max_candidates':8,'fork_max_distance':16.};started=time.time()
    for name in names[a.shard::4]:
        while not (R/('reproduced_'+a.base)/(name+'_score.json')).exists():time.sleep(5)
        raw=json.loads((R/('reproduced_'+a.base)/(name+'.json')).read_text());nodes={int(k):v for k,v in raw['nodes'].items()};edges=raw['edges'];graph_sig=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        feature_sig=hashlib.sha256(json.dumps([signatures,graph_sig,18.,8],sort_keys=True).encode()).hexdigest()[:16];dest=C/f'setfeatures_{name}_{feature_sig}.npz'
        if dest.exists():continue
        ids=sorted(nodes,key=lambda n:(nodes[n]['t'],n));lookup={n:i for i,n in enumerate(ids)};out,prev,frames,pos=structure(nodes,edges);seq,motion=sequences(ids,nodes,out,prev);er,_=candidate_rows(nodes,edges,config,include_forks=False);banks=[]
        for model,signature in zip(models,signatures):
            bankpath=C/f'trackfusion_bank_v3_{name}_{signature}_{graph_sig}.npz'
            if bankpath.exists():
                with np.load(bankpath) as d:bank=d['bank']
            else:
                movie=TrackMovie(TRAIN/(name+'.zarr'),model.channels);crops=np.empty((len(ids),*((model.channels,) if model.channels>1 else ()),8,16,16),np.float16)
                for t,ns in sorted(frames.items()):
                    for start in range(0,len(ns),512):
                        chosen=ns[start:start+512];crops[[lookup[n] for n in chosen]]=movie.crops(t,[[nodes[n][k] for k in ['z','y','x']] for n in chosen])
                bank=encode_bank(model,crops,seq,motion);tmp=bankpath.with_name(bankpath.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,bank=bank);os.replace(tmp,bankpath)
            banks.append(bank)
        positions=np.asarray([pos[n] for n in ids]);mf=FastMotionFeatures(name,nodes,edges,TRAIN,C);parts=[]
        for start in range(0,len(er),4096):
            rr=er[start:start+4096];ix=np.asarray([[lookup[int(n)] for n in row] for row in rr]);g=geometries(ix,'edge',positions,seq,motion);m=mf.rows('edge',rr)
            ff=[fusion_features(model,bank,ix,invariant_features(m,'edge') if model.edge_dim==338 else g,m,'edge') for model,bank in zip(models,banks)]
            parts.append(np.concatenate([ff[0]]+[f[:,338:] for f in ff[1:]],1))
        features=np.concatenate(parts);tmp=dest.with_name(dest.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,rows=er,features=features);os.replace(tmp,dest)
        print('UNLABELLED_ENCODINGS_READY',name,round(time.time()-started),flush=True)
    (R/f'audit_feature_precompute_{a.base}_{a.group}_{a.encoder_group}_{a.shard}.json').write_text(json.dumps({'shard':a.shard,'annotations_read':False,'candidate_heads_run':False,'metric_run':False,'encoders_sha256':signatures,'seconds':time.time()-started}))
