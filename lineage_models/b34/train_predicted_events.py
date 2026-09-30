from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse, json, time, random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from cell_event import CellEventNet

def atomic_save(checkpoint,path):
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    torch.save(checkpoint,temporary)
    os.replace(temporary,path)

class Events(Dataset):
    def __init__(self,root,names,kind):
        self.root=Path(root);self.kind=kind;self.names=names;self.banks={};self.parts=[];self.geom=[];self.groups=[];self.phase=[]
        for movie,name in enumerate(names):
            with np.load(self.root/(name+'.npz')) as d:
                r=d['edges' if kind=='edge' else 'forks'];g=d['edge_geom' if kind=='edge' else 'fork_geom']
                self.parts.append(np.c_[np.full(len(r),movie),r]);self.geom.append(g);self.phase.append(d['phase'])
        self.rows=np.concatenate(self.parts).astype(np.int64);self.geom=np.concatenate(self.geom).astype(np.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];movie=int(r[0]);ids=r[1:-1]
        if movie not in self.banks:self.banks[movie]=np.load(self.root/(self.names[movie]+'_patches.npy'),mmap_mode='r')
        return np.array(self.banks[movie][ids]),self.geom[i].copy(),np.float32(r[-1]),self.phase[movie][ids]

def augment(x,g,kind,shared_photometry=False,time_jitter=False,spatial_jitter_um=None):
    # Shared orientation for all cells in an event; independent intensity noise.
    for axis in [-1,-2,-3]:
        if random.random()<.5:
            x=x.flip(axis)
            if kind=='edge':
                j=axis+3
                for start in [0,4,7,10]:g[:,start+j]*=-1
    if random.random()<.5:
        x=x.transpose(-1,-2)
        if kind=='edge':
            for start in [0,4,7,10]:g[:,[start+1,start+2]]=g[:,[start+2,start+1]]
    # Detector-to-GT displacement augmentation is applied to each cell crop.
    limits=(1,1,1) if spatial_jitter_um is None else tuple(int(round(spatial_jitter_um/s)) for s in [1.625,.8125,.8125])
    for cell in range(x.shape[1]):
        shift=tuple(random.randint(-limit,limit) for limit in limits)
        x[:,cell]=torch.roll(x[:,cell],shift,(-3,-2,-1))
    if shared_photometry:
        x=x*torch.empty((len(x),1,1,1,1,1),device=x.device).uniform_(.65,1.4)
        x=x*torch.empty((len(x),x.shape[1],1,1,1,1),device=x.device).uniform_(.9,1.1)
    else:x=x*torch.empty((len(x),x.shape[1],1,1,1,1),device=x.device).uniform_(.65,1.4)
    if time_jitter and kind=='fork' and random.random()<.25:
        offset=random.choice([-1,1]);ix=(torch.arange(x.shape[2],device=x.device)+offset).clamp(0,x.shape[2]-1)
        x=x[:,:,ix]
    x=x+torch.randn_like(x)*random.uniform(0,.04)
    # Tracking centroids and local velocities are noisy at inference time.
    if kind=='edge':g[:,:14]+=torch.randn_like(g[:,:14])*.035
    else:g[:,:25]+=torch.randn_like(g[:,:25])*.025
    return x,g

@torch.inference_mode()
def validate(model,root,names,device):
    model.eval();cache={};all_results={}
    for name in names:
        bank=np.load(Path(root)/(name+'_patches.npy'),mmap_mode='r');chunks=[]
        for start in range(0,len(bank),384):
            x=torch.from_numpy(np.array(bank[start:start+384])).to(device,dtype=torch.float32)
            with torch.autocast('cuda',dtype=torch.float16):chunks.append(model.encode(x).float().cpu())
        cache[name]=torch.cat(chunks).to(device)
    for kind in ['edge','fork']:
        yy=[];pp=[];gg=[]
        for name in names:
            with np.load(Path(root)/(name+'.npz')) as data:
                rows=data['edges' if kind=='edge' else 'forks'];geom=data['edge_geom' if kind=='edge' else 'fork_geom']
            emb=cache[name]
            for start in range(0,len(rows),4096):
                r=rows[start:start+4096];g=torch.from_numpy(geom[start:start+4096]).to(device)
                es=[emb[r[:,i]] for i in range(r.shape[1]-1)]
                logits=model.edge_logits(*es,g) if kind=='edge' else model.fork_logits(*es,g)
                yy.extend(r[:,-1]);pp.extend(logits.sigmoid().cpu().numpy());gg.extend([name]*len(r))
        y=np.asarray(yy);p=np.asarray(pp)
        all_results[kind]={'ap':float(average_precision_score(y,p)),'auc':float(roc_auc_score(y,p)),
                           'logloss':float(log_loss(y,np.clip(p,1e-6,1-1e-6))),'positive':int(y.sum()),'n':len(y)}
        all_results[kind+'_predictions']={'y':y,'prob':p,'movie':np.asarray(gg)}
    return all_results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='/workspace/biohub/events');p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20260921);p.add_argument('--context',type=int,default=3);p.add_argument('--width',type=int,default=24)
    p.add_argument('--seconds',type=int,default=2700);p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--resume');p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--shared-photometry',action='store_true');p.add_argument('--time-jitter',action='store_true');p.add_argument('--initialize')
    p.add_argument('--spatial-jitter-um',type=float)
    p.add_argument('--distill-weight',type=float,default=.05)
    p.add_argument('--hard-negative-weight',type=float,default=10.)
    p.add_argument('--architecture',choices=['residual','pretrained_unet'],default='residual')
    p.add_argument('--schedule',choices=['wall','steps'],default='steps');args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    split=json.loads((Path(args.data)/'split.json').read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    output_lock=(out/'.training.lock').open('a')
    fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'history.json').exists():raise RuntimeError('Output already contains training history; use a fresh directory')
    train={k:Events(args.data,split['train'],k) for k in ['edge','fork']}
    loaders={}
    for k,d in train.items():
        y=d.rows[:,-1];pos=float(y.sum());neg=float(len(y)-pos)
        # Natural edge prior, 25% mitosis events in fork batches.
        weights=np.where(y>0,(neg/max(pos,1))*(1/3 if k=='fork' else .6),1.)
        if k=='fork':
            positive_sources={tuple(r[:2]) for r in d.rows if r[-1]}
            hard=np.array([tuple(r[:2]) in positive_sources and r[-1]==0 for r in d.rows])
            weights[hard]=args.hard_negative_weight
            weights[y>0]=weights[y==0].sum()/max(pos,1)/3
            print('PREDICTED_HARD_NEGATIVES',int(hard.sum()),flush=True)
        sample=WeightedRandomSampler(torch.from_numpy(weights),num_samples=max(len(d),args.batch*100),replacement=True)
        loaders[k]=DataLoader(d,batch_size=args.batch if k=='edge' else args.batch//2,sampler=sample,
            num_workers=args.workers,pin_memory=True,persistent_workers=True,drop_last=True)
    config={'seed':args.seed,'context':args.context,'width':args.width,'normalization_context':3}
    if args.architecture=='pretrained_unet':
        from pretrained_event import PretrainedEventNet
        config['architecture']='pretrained_unet';model=PretrainedEventNet(**config,initialize=True).cuda()
    else:model=CellEventNet(**config).cuda()
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert checkpoint['config']==config,(checkpoint['config'],config)
        model.load_state_dict(checkpoint['model'])
    if args.initialize:
        checkpoint=torch.load(args.initialize,map_location='cpu',weights_only=False);target=model.state_dict();old=checkpoint['model']
        for key,value in old.items():
            if target[key].shape==value.shape:target[key]=value.clone()
        key='encoder.0.weight';target[key].zero_();offset=(args.context-checkpoint['config']['context'])//2
        target[key][:,offset:offset+checkpoint['config']['context']]=old[key]
        key='photometry.0.weight';target[key].zero_();old_context=checkpoint['config']['context']
        for half in [0,1]:target[key][:,half*args.context+offset:half*args.context+offset+old_context]=old[key][:,half*old_context:(half+1)*old_context]
        model.load_state_dict(target)
    from cell_event import load_event_model
    teacher,_=load_event_model('/workspace/biohub/B1/best.pt')
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.015)
    iters={k:iter(v) for k,v in loaders.items()};started=time.time();best=-1.;history=[]
    print('TRAIN_START',json.dumps({'config':config,'examples':{k:len(v) for k,v in train.items()},'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    for step in range(1,args.steps+1):
        if time.time()-started>args.seconds:break
        kind='fork' if step%3==0 else 'edge'
        try:batch=next(iters[kind])
        except StopIteration:iters[kind]=iter(loaders[kind]);batch=next(iters[kind])
        x,g,y,phase=[b.cuda(non_blocking=True).float() for b in batch]
        x,g=augment(x,g,kind,args.shared_photometry,args.time_jitter,args.spatial_jitter_um);model.train();optimizer.zero_grad(set_to_none=True)
        progress=min(1.,max(step/args.steps if args.schedule=='steps' else 0.,(time.time()-started)/args.seconds))
        lr=args.lr*(.15+.85*(1+np.cos(np.pi*progress))/2)*min(1.,step/100)
        for group in optimizer.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model.encode(x.flatten(0,1)).reshape(len(x),x.shape[1],-1)
            logits=model.edge_logits(z[:,0],z[:,1],g) if kind=='edge' else model.fork_logits(z[:,0],z[:,1],z[:,2],g)
            bce=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')
            pt=torch.where(y>.5,logits.sigmoid(),1-logits.sigmoid()).float()
            loss=(bce*(1-pt).clamp_min(.1)).mean()
            with torch.no_grad():
                tz=teacher.encode(x.flatten(0,1)).reshape(len(x),x.shape[1],-1)
                tl=teacher.edge_logits(tz[:,0],tz[:,1],g) if kind=='edge' else teacher.fork_logits(tz[:,0],tz[:,1],tz[:,2],g)
            loss=loss+args.distill_weight*F.binary_cross_entropy_with_logits(logits.float(),tl.float().sigmoid())
            phase_logits=model.phase(z).squeeze(-1)
            phase_mask=phase>=0
            if phase_mask.any():loss=loss+.08*F.binary_cross_entropy_with_logits(phase_logits.float()[phase_mask],phase[phase_mask])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
        if step%50==0:print('STEP',step,kind,'loss',round(float(loss),5),'seconds',round(time.time()-started),'lr',lr,flush=True)
        if step%2000==0:
            result=validate(model,args.data,split['calibration'],'cuda')
            compact={k:v for k,v in result.items() if not k.endswith('_predictions')};compact.update(step=step,seconds=time.time()-started)
            objective=compact['edge']['ap']+.35*compact['fork']['ap'];compact['objective']=objective;history.append(compact)
            print('VALIDATION',json.dumps(compact),flush=True)
            ckpt={'config':config,'model':{k:v.cpu() for k,v in model.state_dict().items()},'step':step,'validation':compact,'split':split,'training_arguments':vars(args)}
            atomic_save(ckpt,out/'last.pt')
            atomic_save(ckpt,out/('step_'+str(step)+'.pt'))
            if objective>best:
                best=objective;atomic_save(ckpt,out/'best.pt')
                for kind in ['edge','fork']:np.savez_compressed(out/(kind+'_calibration.npz'),**result[kind+'_predictions'])
            temporary=out/('history.'+str(os.getpid())+'.tmp')
            temporary.write_text(json.dumps(history,indent=2));os.replace(temporary,out/'history.json')
    if not (out/'best.pt').exists():
        raise RuntimeError('Training ended before the first independent calibration pass')
    print('TRAIN_DONE',step,round(time.time()-started),best,flush=True)
