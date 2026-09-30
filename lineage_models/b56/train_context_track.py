from pathlib import Path
import os,json,time,argparse,hashlib
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np,torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from context_track import ContextTrackNet
R=Path('/workspace/biohub');D=R/'track_context7'
def load(names):
    features=[];motion=[];geoms={k:[] for k in ['edge','fork']};rows={k:[] for k in geoms};weights={k:[] for k in geoms};offset=0
    for name in names:
        latent=np.load(D/(name+'_latent.npy'))
        with np.load(D/(name+'.npz')) as d:
            features.append(latent[d['seq']]);motion.append(d['motion'])
            for kind in rows:
                rr=d[kind].copy();rr[:,:-1]+=offset;rows[kind].append(rr);geoms[kind].append(d[kind+'_geom']);weights[kind].append(d[kind+'_weight'] if kind+'_weight' in d else np.ones(len(rr),np.float32))
        offset+=len(latent)
    banks=torch.from_numpy(np.concatenate(features,axis=1)).cuda();moves=torch.from_numpy(np.concatenate(motion,axis=1)).cuda()
    return banks,moves,{k:(torch.from_numpy(np.concatenate(rows[k])).cuda(),torch.from_numpy(np.concatenate(geoms[k])).cuda(),torch.from_numpy(np.concatenate(weights[k])).cuda()) for k in rows}
def forward(model,bank,moves,rows,geometry,kind,augment=False):
    rr=rows[:,:-1];z=torch.stack([bank[int(j>0),rr[:,j]] for j in range(rr.shape[1])],1).float();m=torch.stack([moves[int(j>0),rr[:,j]] for j in range(rr.shape[1])],1).clone();g=geometry.clone()
    if augment:
        z=z+torch.randn_like(z)*.03;m[...,:3]+=torch.randn_like(m[...,:3])*.025
        keep=torch.randint(2,11,(len(z),z.shape[1],1),device='cuda');m[...,3]*=(torch.arange(10,device='cuda')[None,None]<keep)
        sel=torch.rand(len(g),device='cuda')<.7
        for offset in ([0] if kind=='edge' else [434,482]):g[sel,offset+44]=.5;g[sel,offset+45]=0.
    b,k,t,w=z.shape
    with torch.autocast('cuda',dtype=torch.float16):zz=model.encode_track(z.reshape(b*k,t,w),m.reshape(b*k,t,4)).reshape(b,k,w);logits=model.classify(zz,g,kind)
    return logits.float()
@torch.inference_mode()
def evaluate(model,data):
    model.eval();bank,moves,sets=data;result={}
    for kind,(rows,g,_) in sets.items():
        pp=[]
        for start in range(0,len(rows),512):pp.append(forward(model,bank,moves,rows[start:start+512],g[start:start+512],kind).sigmoid().cpu().numpy())
        result[kind]={'ap':float(average_precision_score(rows[:,-1].cpu().numpy(),np.concatenate(pp)))}
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();O=R/('TrackContext7_'+str(a.seed));O.mkdir(exist_ok=True)
    assert not (O/'history.json').exists();torch.set_num_threads(3);torch.manual_seed(a.seed);torch.backends.mha.set_fastpath_enabled(False)
    split=json.loads((D/'split.json').read_text());train=load(split['train']);cal=load(split['calibration']);initial=torch.load(D/'initial.pt',map_location='cpu',weights_only=False);model=ContextTrackNet(**initial['config']).cuda();model.load_state_dict(initial['model'])
    for p in model.frame.parameters():p.requires_grad=False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,weight_decay=.03);history=[];best=-1;start=time.time();bank,moves,sets=train
    sampling={}
    for k,(rr,g,w) in sets.items():
        y=rr[:,-1];sampling[k]=w*torch.where(y>0,(len(y)-y.sum())/y.sum().clamp_min(1)*(1 if k=='edge' else .5),1.)
    protocol={'train':split['train'],'calibration':split['calibration'],'new_audit_excluded':split['new_audit'],'initialization':'Frozen inherited seven-frame B4 centroid scene encoder with newly fitted ten-step track attention and physical-motion event heads; inherited pretraining overlaps held-out sets.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'encoder_sha256':json.loads((D/'ready.json').read_text())['inherited_encoder_sha256']}
    (O/'training_protocol.json').write_text(json.dumps(protocol,indent=2))
    for step in range(1,12001):
        kind='fork' if step%2==0 else 'edge';rows,g,w=sets[kind];ix=torch.multinomial(sampling[kind],256,replacement=True);model.train();optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr']=5e-4*(.1+.9*(1+np.cos(np.pi*step/12000))/2)*min(1,step/100)
        logits=forward(model,bank,moves,rows[ix],g[ix],kind,True);loss=F.binary_cross_entropy_with_logits(logits,rows[ix,-1].float());loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.);optimizer.step()
        if step%200==0:print('CONTEXT_TRAIN',step,float(loss.detach()),round(time.time()-start),flush=True)
        if step%1200==0:
            metrics=evaluate(model,cal);row={'step':step,'metrics':metrics,'seconds':time.time()-start};history.append(row);value=metrics['edge']['ap']+.5*metrics['fork']['ap']
            if value>best:best=value;torch.save({'config':initial['config'],'model':model.state_dict(),'protocol':protocol,'metrics':metrics,'step':step},O/'best.pt')
            (O/'history.json').write_text(json.dumps(history,indent=2));print('CONTEXT_VALIDATED',json.dumps(row),flush=True)
    (O/'ready.json').write_text(json.dumps({'seconds':time.time()-start}));print('CONTEXT_TRACK_DONE',flush=True)
