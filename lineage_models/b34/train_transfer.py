from pathlib import Path
import os,json,time,random,argparse,fcntl
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from transfer_model import TransferLineageNet
from train_events import atomic_save

class TransferEvents(Dataset):
    def __init__(self,root,names,kind):
        self.root=Path(root);self.names=names;self.kind=kind;self.banks={};rows=[];labels=[];geom=[]
        for m,name in enumerate(names):
            with np.load(self.root/(name+'.npz')) as d:
                y=d[kind+'_y'];labels.append(y);geom.append(d[kind+'_geom']);rows.append(np.c_[np.full(len(y),m),d[kind+'_rows']])
        self.rows=np.concatenate(rows).astype(np.int64);self.y=np.concatenate(labels).astype(np.float32);self.geom=np.concatenate(geom).astype(np.float32)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        m,*ix=self.rows[i]
        if m not in self.banks:self.banks[m]=np.load(self.root/(self.names[m]+'_features.npy'),mmap_mode='r')
        return np.array(self.banks[m][ix]),self.geom[i].copy(),self.y[i]

class ForkPairs(Dataset):
    def __init__(self,events):
        self.events=events;positive={};negative={}
        for i,(row,y) in enumerate(zip(events.rows,events.y)):
            key=tuple(row[:2]);(positive if y else negative).setdefault(key,[]).append(i)
        self.groups=[(pp,negative[k]) for k,pp in positive.items() if k in negative]
    def __len__(self):return max(4096,len(self.groups)*32)
    def __getitem__(self,i):
        positive,negative=self.groups[i%len(self.groups)];a=self.events[random.choice(positive)];b=self.events[random.choice(negative)]
        return np.stack([a[0],b[0]]),np.stack([a[1],b[1]]),np.asarray([1,0],np.float32)

@torch.inference_mode()
def validate(model,root,names,batch):
    model.eval();result={}
    for kind in ['edge','fork']:
        ds=TransferEvents(root,names,kind);loader=DataLoader(ds,batch_size=batch,num_workers=2,pin_memory=True);ys=[];ps=[];bp=[]
        for x,g,y in loader:
            with torch.autocast('cuda',dtype=torch.float16):logits,base,_=model(x.cuda().float(),g.cuda().float(),kind,True)
            ys.extend(y.numpy());ps.extend(logits.float().sigmoid().cpu().numpy());bp.extend(base.float().sigmoid().cpu().numpy())
        y=np.asarray(ys);p=np.asarray(ps);assert np.isfinite(p).all()
        result[kind]={'ap':float(average_precision_score(y,p)),'base_ap':float(average_precision_score(y,bp)),'auc':float(roc_auc_score(y,p)),'logloss':float(log_loss(y,np.clip(p,1e-6,1-1e-6))),'positive':int(y.sum()),'n':len(y)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='/workspace/biohub/transfer_data');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261015);p.add_argument('--width',type=int,default=96);p.add_argument('--layers',type=int,default=2);p.add_argument('--steps',type=int,default=16000);p.add_argument('--seconds',type=int,default=3600);p.add_argument('--batch',type=int,default=512);p.add_argument('--regularization',type=float,default=.0005);p.add_argument('--hard-negative-weight',type=float,default=1.);p.add_argument('--pairwise-weight',type=float,default=0.);p.add_argument('--initialize');a=p.parse_args()
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    root=Path(a.data);out=Path(a.out);out.mkdir(exist_ok=True);lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'history.json').exists():raise RuntimeError('Refusing to overwrite an existing training run')
    while not (root/'ready.json').exists():time.sleep(10)
    split=json.loads((root/'split.json').read_text());assert not set(split['train'])&set(split['calibration']+split['audit']);train={k:TransferEvents(root,split['train'],k) for k in ['edge','fork']};loaders={}
    for kind,ds in train.items():
        pos=ds.y.sum();neg=len(ds)-pos;w=np.where(ds.y>0,neg/max(pos,1)*(.5 if kind=='edge' else .333),1)
        if kind=='fork' and a.hard_negative_weight>1:
            positive_sources={tuple(r[:2]) for r,y in zip(ds.rows,ds.y) if y};hard=np.array([tuple(r[:2]) in positive_sources and y==0 for r,y in zip(ds.rows,ds.y)])
            w[hard]=a.hard_negative_weight;w[ds.y>0]=w[ds.y==0].sum()/max(pos,1)*.333
            print('SAME_PARENT_HARD_NEGATIVES',int(hard.sum()),'weight',a.hard_negative_weight,flush=True)
        sampler=WeightedRandomSampler(torch.from_numpy(w),max(len(ds),a.batch*100),replacement=True)
        loaders[kind]=DataLoader(ds,batch_size=a.batch,sampler=sampler,num_workers=3,persistent_workers=True,pin_memory=True,drop_last=True)
    base=torch.load('/workspace/biohub/B1/best.pt',map_location='cpu',weights_only=False)
    iters={k:iter(v) for k,v in loaders.items()};config={'width':a.width,'layers':a.layers,'base_config':base['config']}
    initial=torch.load(a.initialize,map_location='cpu',weights_only=False) if a.initialize else None
    if initial:config=initial['config']
    model=TransferLineageNet(**config).cuda();model.base.load_state_dict(base['model'])
    if initial:model.load_state_dict(initial['model'])
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.03)
    pairloader=None
    if a.pairwise_weight>0:
        pairloader=DataLoader(ForkPairs(train['fork']),batch_size=64,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=True,drop_last=True);pairiter=iter(pairloader)
    history=[];best=-1.;start=time.time();print('TRANSFER_TRAIN_START',json.dumps({'config':config,'arguments':vars(a),'examples':{k:len(v) for k,v in train.items()},'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}),flush=True)
    for step in range(1,a.steps+1):
        if time.time()-start>a.seconds:break
        kind='fork' if step%2==0 else 'edge'
        try:x,g,y=next(iters[kind])
        except StopIteration:iters[kind]=iter(loaders[kind]);x,g,y=next(iters[kind])
        pair_count=0
        if kind=='fork' and pairloader is not None:
            try:px,pg,py=next(pairiter)
            except StopIteration:pairiter=iter(pairloader);px,pg,py=next(pairiter)
            pair_count=len(px);x=torch.cat([x,px.flatten(0,1)]);g=torch.cat([g,pg.flatten(0,1)]);y=torch.cat([y,py.flatten(0,1)])
        x,g,y=x.cuda().float(),g.cuda().float(),y.cuda();x=x+torch.randn_like(x)*random.uniform(0,.04);g=g+torch.randn_like(g)*.02
        if random.random()<.15:x[:,:,random.choice([0,2])]=x[:,:,1]
        model.train();opt.zero_grad(set_to_none=True)
        progress=max(step/a.steps,(time.time()-start)/a.seconds);lr=2e-4*(.15+.85*(1+np.cos(min(1,progress)*np.pi))/2)*min(1,step/100)
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,base,delta=model(x,g,kind,True);bce=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none');pt=torch.where(y>.5,logits.sigmoid(),1-logits.sigmoid()).float();loss=(bce*(1-pt).clamp_min(.1)).mean()+a.regularization*delta.float().square().mean()
            if pair_count:
                paired=logits[-2*pair_count:].float().reshape(-1,2);loss=loss+a.pairwise_weight*F.softplus(2.+paired[:,1]-paired[:,0]).mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if step%100==0:print('TRANSFER_STEP',step,kind,'loss',float(loss.detach()),'seconds',round(time.time()-start),flush=True)
        if step%1000==0:
            stats=validate(model,root,split['calibration'],a.batch);objective=stats['edge']['ap']+.45*stats['fork']['ap'];stats.update(step=step,seconds=round(time.time()-start),objective=objective);history.append(stats)
            ckpt={'config':config,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step,'validation':stats,'split':split,'training_arguments':vars(a)};atomic_save(ckpt,out/'last.pt')
            if objective>best:best=objective;atomic_save(ckpt,out/'best.pt')
            if step%4000==0:atomic_save(ckpt,out/('step_'+str(step)+'.pt'))
            (out/'history.json').write_text(json.dumps(history,indent=2));print('TRANSFER_VALIDATION',json.dumps(stats),flush=True)
    assert (out/'best.pt').exists();(out/'done.json').write_text(json.dumps({'step':step,'seconds':time.time()-start,'best_objective':best}));print('TRANSFER_TRAIN_DONE',step,best,flush=True)
