from pathlib import Path
import os,json,time,argparse,random,fcntl
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from joint_model import JointLineageNet,SHAPE,SPACING
from train_events import atomic_save

class JointEvents(Dataset):
    def __init__(self,root,names,kind):
        self.root=Path(root);self.names=names;self.kind=kind;self.banks={};self.parts=[];self.geom=[];self.relative=[];self.child=[];self.phase=[]
        for m,name in enumerate(names):
            with np.load(self.root/(name+'.npz')) as d:
                r=d[kind+'_rows'];self.parts.append(np.c_[np.full(len(r),m),r]);self.geom.append(d[kind+'_geom']);self.relative.append(d[kind+'_rel']);self.child.append(d['children']);self.phase.append(d['phase'])
        self.rows=np.concatenate(self.parts).astype(np.int64);self.geom=np.concatenate(self.geom).astype(np.float32);self.relative=np.concatenate(self.relative).astype(np.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        m,s,y=self.rows[i]
        if m not in self.banks:self.banks[m]=np.load(self.root/(self.names[m]+'_patches.npy'),mmap_mode='r')
        return np.array(self.banks[m][s]),self.relative[i].copy(),self.geom[i].copy(),np.float32(y),self.child[m][s].copy(),np.float32(self.phase[m][s])

class JointForkPairs(Dataset):
    def __init__(self,events):
        self.events=events;positive={};negative={}
        for i,r in enumerate(events.rows):(positive if r[-1] else negative).setdefault(tuple(r[:2]),[]).append(i)
        self.groups=[(v,negative[k]) for k,v in positive.items() if k in negative]
    def __len__(self):return max(4096,len(self.groups)*32)
    def __getitem__(self,i):
        pp,nn=self.groups[i%len(self.groups)];a=self.events[random.choice(pp)];b=self.events[random.choice(nn)]
        return tuple(np.stack([x,y]) for x,y in zip(a,b))

def augment(x,relative,g,children,kind,spatial_jitter=0.):
    for j,axis in enumerate([-3,-2,-1]):
        if random.random()<.5:
            x=x.flip(axis);relative[:,:,j]=-relative[:,:,j]-float(SPACING[j]);children[:,:,j]=-children[:,:,j]-float(SPACING[j])
            if kind=='edge':
                for start in [0,4,7,10]:g[:,start+j]*=-1
    if random.random()<.5:
        x=x.transpose(-1,-2);relative[:,:,[1,2]]=relative[:,:,[2,1]];children[:,:,[1,2]]=children[:,:,[2,1]]
        if kind=='edge':
            for start in [0,4,7,10]:g[:,[start+1,start+2]]=g[:,[start+2,start+1]]
    # Whole-scene photometric augmentation preserves relative daughter brightness.
    x=x*torch.empty((len(x),1,1,1,1),device=x.device).uniform_(.6,1.5)
    x=x*torch.empty((len(x),x.shape[1],1,1,1),device=x.device).uniform_(.93,1.07)
    x=x+torch.randn_like(x)*random.uniform(0,.035)
    if spatial_jitter>0:
        shifts=[random.randint(-int(round(spatial_jitter/s)),int(round(spatial_jitter/s))) for s in SPACING]
        x=torch.roll(x,shifts,(-3,-2,-1))
        delta=x.new_tensor(np.asarray(shifts)*SPACING);relative+=delta;children+=delta
    relative[:,1:]+=torch.randn_like(relative[:,1:])*.4
    g+=torch.randn_like(g)*.02
    return x,relative,g,children

def dense_loss(logits,children):
    device=logits.device
    axes=[(torch.arange(n,device=device)-n//2)*float(s) for n,s in zip(SHAPE,SPACING)]
    grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
    valid=children.isfinite().all(-1)
    distance=((grid[None,None]-torch.nan_to_num(children)[:,:,None,None,None,:])/1.4).square().sum(-1)
    target=(torch.exp(-.5*distance)*valid[:,:,None,None,None]).amax(1)
    if os.environ.get('JOINT_DENSE_MODE','focal')=='softmax':
        target=target/(target.sum((1,2,3),keepdim=True)+1e-6)
        lp=F.log_softmax(logits[:,0].float().flatten(1),dim=1).reshape_as(target)
        return -(target*lp).sum((1,2,3)).mean()
    p=logits[:,0].float().sigmoid().clamp(1e-5,1-1e-5)
    positive=-target*(1-p).square()*p.log()
    negative=-.012*(1-target).pow(4)*p.square()*(1-p).log()
    return ((positive+negative).sum((1,2,3))/(target.sum((1,2,3))+1)).mean()

@torch.inference_mode()
def validate(model,data,names,batch):
    model.eval();result={}
    for kind in ['edge','fork']:
        ds=JointEvents(data,names,kind);loader=DataLoader(ds,batch_size=batch,num_workers=4,pin_memory=True)
        ys=[];ps=[];localization=[]
        for x,r,g,y,c,phase in loader:
            with torch.autocast('cuda',dtype=torch.float16):
                logits,heat,_=model(x.cuda().float(),r.cuda(),g.cuda(),kind)
            ys.extend(y.numpy());ps.extend(logits.float().sigmoid().cpu().numpy())
            if kind=='edge':
                valid=(y.numpy()>.5)&np.isfinite(c.numpy()[:,0]).all(-1)&~np.isfinite(c.numpy()[:,1]).all(-1)
                if valid.any():
                    ix=heat[:,0].float().flatten(1).argmax(1).cpu().numpy()[valid]
                    predicted=(np.stack(np.unravel_index(ix,SHAPE),-1)-np.asarray(SHAPE)//2)*SPACING
                    localization.extend(np.linalg.norm(predicted-c.numpy()[valid,0],axis=-1).tolist())
        y=np.asarray(ys);pr=np.asarray(ps)
        assert np.isfinite(pr).all()
        result[kind]={'ap':float(average_precision_score(y,pr)),'auc':float(roc_auc_score(y,pr)),'logloss':float(log_loss(y,np.clip(pr,1e-6,1-1e-6))),'n':len(y),'positive':int(y.sum())}
        if kind=='edge' and localization:
            loc=np.asarray(localization);result['localization']={'n':len(loc),'median_um':float(np.median(loc)),'mean_um':float(loc.mean()),'recall_3um':float((loc<=3).mean()),'recall_5um':float((loc<=5).mean())}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='/workspace/biohub/joint_data');p.add_argument('--out',required=True);p.add_argument('--context',type=int,default=7);p.add_argument('--width',type=int,default=24);p.add_argument('--architecture',default='joint',choices=['joint','temporal_attention']);p.add_argument('--seed',type=int,default=20260923);p.add_argument('--steps',type=int,default=14000);p.add_argument('--seconds',type=int,default=7200);p.add_argument('--batch',type=int,default=64);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--validate-every',type=int,default=1500);p.add_argument('--initialize');a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    root=Path(a.data);out=Path(a.out);out.mkdir(exist_ok=True,parents=True);lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'history.json').exists():raise RuntimeError('Refusing to overwrite an existing run')
    while not (root/'ready.json').exists():time.sleep(10)
    split=json.loads((root/'split.json').read_text());assert not set(split['train'])&set(split['calibration']+split['audit'])
    a.hard_negative_weight=float(os.environ.get('JOINT_HARD_NEGATIVE_WEIGHT','1'));a.pairwise_weight=float(os.environ.get('JOINT_PAIRWISE_WEIGHT','0'))
    train={k:JointEvents(root,split['train'],k) for k in ['edge','fork']};loaders={}
    for k,ds in train.items():
        y=ds.rows[:,-1];pos=y.sum();neg=len(y)-pos;w=np.where(y>0,neg/max(1,pos)*(.5 if k=='edge' else .333),1.)
        if k=='fork' and a.hard_negative_weight>1:
            positive_sources={tuple(r[:2]) for r in ds.rows if r[-1]};hard=np.array([tuple(r[:2]) in positive_sources and r[-1]==0 for r in ds.rows])
            w[hard]=a.hard_negative_weight;w[y>0]=w[y==0].sum()/max(1,pos)*.333;print('JOINT_HARD_NEGATIVES',int(hard.sum()),flush=True)
        sampler=WeightedRandomSampler(torch.from_numpy(w),max(len(y),a.batch*100),replacement=True)
        loaders[k]=DataLoader(ds,batch_size=a.batch if k=='edge' else a.batch//2,sampler=sampler,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
    condition=os.environ.get('JOINT_PARENT_CONDITION','0')=='1';spatial_jitter=float(os.environ.get('JOINT_SPATIAL_JITTER_UM','0'))
    a.parent_condition=condition;a.spatial_jitter_um=spatial_jitter;a.dense_mode=os.environ.get('JOINT_DENSE_MODE','focal');a.dense_weight=float(os.environ.get('JOINT_DENSE_WEIGHT','.18'))
    config={'context':a.context,'width':a.width,'architecture':a.architecture,'parent_condition':condition,'robust_pool':os.environ.get('JOINT_ROBUST_POOL','0')=='1'};model=JointLineageNet(**config).cuda()
    if a.initialize:
        old=torch.load(a.initialize,map_location='cpu',weights_only=False);state=model.state_dict()
        for k,v in old['model'].items():
            if state[k].shape==v.shape:state[k]=v
            elif k=='stem.0.weight' and state[k].shape[1]==v.shape[1]+1:
                state[k].zero_();state[k][:,:v.shape[1]]=v
            else:raise RuntimeError('Incompatible initialization tensor: '+k)
        model.load_state_dict(state)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.02);iters={k:iter(v) for k,v in loaders.items()};history=[];best=-1;best_fork=-1;started=time.time()
    pairloader=None
    if a.pairwise_weight>0:
        pairloader=DataLoader(JointForkPairs(train['fork']),batch_size=max(8,a.batch//4),shuffle=True,num_workers=2,persistent_workers=True,pin_memory=True,drop_last=True);pairiter=iter(pairloader)
    print('JOINT_TRAIN_START',json.dumps({'config':config,'arguments':vars(a),'parameters':sum(p.numel() for p in model.parameters()),'examples':{k:len(d) for k,d in train.items()}}),flush=True)
    for step in range(1,a.steps+1):
        if time.time()-started>a.seconds:break
        kind='fork' if step%3==0 else 'edge'
        try:batch=next(iters[kind])
        except StopIteration:iters[kind]=iter(loaders[kind]);batch=next(iters[kind])
        pair_count=0
        if kind=='fork' and pairloader is not None:
            try:paired_batch=next(pairiter)
            except StopIteration:pairiter=iter(pairloader);paired_batch=next(pairiter)
            pair_count=len(paired_batch[0]);batch=[torch.cat([regular,paired.flatten(0,1)]) for regular,paired in zip(batch,paired_batch)]
        x,r,g,y,c,phase=[b.cuda(non_blocking=True).float() for b in batch];x,r,g,c=augment(x,r,g,c,kind,spatial_jitter)
        model.train();opt.zero_grad(set_to_none=True)
        progress=min(1,max(step/a.steps,(time.time()-started)/a.seconds));lr=a.lr*(.1+.9*(1+np.cos(progress*np.pi))/2)*min(1,step/150)
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,heat,ph=model(x,r,g,kind)
            bce=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none');pt=torch.where(y>.5,logits.sigmoid(),1-logits.sigmoid()).float()
            cls=(bce*(1-pt).clamp_min(.1)).mean();dense=dense_loss(heat,c)
            loss=cls+a.dense_weight*dense+.06*F.binary_cross_entropy_with_logits(ph.float(),phase)
            if pair_count:
                pair_logits=logits[-2*pair_count:].float().reshape(-1,2);loss=loss+a.pairwise_weight*F.softplus(2.+pair_logits[:,1]-pair_logits[:,0]).mean()
        assert torch.isfinite(loss),'Nonfinite loss'
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if step%50==0:print('JOINT_STEP',step,kind,'loss',round(float(loss),5),'class',round(float(cls),5),'dense',round(float(dense),5),'seconds',round(time.time()-started),flush=True)
        if step%a.validate_every==0:
            stats=validate(model,root,split['calibration'],a.batch);objective=stats['edge']['ap']+.45*stats['fork']['ap']
            if a.dense_mode=='softmax':objective=stats['localization']['recall_5um']+.5*stats['localization']['recall_3um']-.02*stats['localization']['median_um']
            stats.update(step=step,seconds=round(time.time()-started),objective=objective);history.append(stats)
            checkpoint={'config':config,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step,'validation':stats,'split':split,'training_arguments':vars(a)}
            atomic_save(checkpoint,out/'last.pt')
            if objective>best:best=objective;atomic_save(checkpoint,out/'best.pt')
            if stats['fork']['ap']>best_fork:best_fork=stats['fork']['ap'];atomic_save(checkpoint,out/'best_fork.pt')
            if step%3000==0:atomic_save(checkpoint,out/('step_'+str(step)+'.pt'))
            (out/'history.json').write_text(json.dumps(history,indent=2));print('JOINT_VALIDATION',json.dumps(stats),flush=True)
    if not (out/'best.pt').exists():raise RuntimeError('No validated checkpoint produced')
    print('JOINT_TRAIN_DONE',step,round(time.time()-started),best,flush=True)
