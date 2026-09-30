from pathlib import Path
import os,json,time,random,argparse,fcntl
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from trajectory_model import TrajectoryNet
from train_events import atomic_save

class Trajectories(Dataset):
    def __init__(self,root,names,kind):
        self.root=Path(root);self.names=names;self.kind=kind;self.banks={};rows=[];labels=[];geom=[]
        for m,name in enumerate(names):
            with np.load(self.root/(name+'.npz')) as d:
                y=d[kind+'_y'];labels.append(y);geom.append(d[kind+'_geom']);rows.append(np.c_[np.full(len(y),m),np.arange(len(y))])
        self.rows=np.concatenate(rows).astype(np.int64);self.y=np.concatenate(labels).astype(np.float32);self.geom=np.concatenate(geom).astype(np.float32)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        m,j=self.rows[i]
        if m not in self.banks:self.banks[m]=np.load(self.root/(self.names[m]+'_'+self.kind+'_x.npy'),mmap_mode='r')
        return np.array(self.banks[m][j]),self.geom[i].copy(),self.y[i]

def augment(x,g):
    gain=torch.empty((len(x),1,1,1),device=x.device).uniform_(.65,1.4)*torch.empty((len(x),x.shape[1],1,1),device=x.device).uniform_(.9,1.1)
    x[:,:,:,:15]*=gain;x[:,:,:,21:22]*=gain
    x[:,:,:,:24]+=torch.randn_like(x[:,:,:,:24])*.01
    if random.random()<.5:
        end=random.randint(2,6);x[:,:,end:]=x[:,:,end-1:end];x[:,:,end:,27]=0.;x[:,:,:,28]=torch.minimum(x[:,:,:,28],x.new_tensor(end/8))
    g=g+torch.randn_like(g)*.03
    return x,g

@torch.inference_mode()
def validate(model,root,names,batch):
    model.eval();result={}
    for kind in ['edge','fork']:
        ds=Trajectories(root,names,kind);loader=DataLoader(ds,batch_size=batch,num_workers=2,pin_memory=True);ys=[];ps=[]
        for x,g,y in loader:
            with torch.autocast('cuda',dtype=torch.float16):logits=model(x.cuda().float(),g.cuda().float(),kind)
            ys.extend(y.numpy());ps.extend(logits.float().sigmoid().cpu().numpy())
        y=np.asarray(ys);p=np.asarray(ps);assert np.isfinite(p).all()
        result[kind]={'ap':float(average_precision_score(y,p)),'auc':float(roc_auc_score(y,p)),'logloss':float(log_loss(y,np.clip(p,1e-6,1-1e-6))),'positive':int(y.sum()),'n':len(y)}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='/workspace/biohub/trajectory_data');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261005);p.add_argument('--width',type=int,default=96);p.add_argument('--layers',type=int,default=2);p.add_argument('--steps',type=int,default=20000);p.add_argument('--seconds',type=int,default=5400);p.add_argument('--batch',type=int,default=512);a=p.parse_args()
    torch.set_num_threads(3);torch.backends.mha.set_fastpath_enabled(False);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    root=Path(a.data);out=Path(a.out);out.mkdir(exist_ok=True);lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'history.json').exists():raise RuntimeError('Refusing to overwrite an existing training run')
    while not (root/'ready.json').exists():time.sleep(10)
    split=json.loads((root/'split.json').read_text());train={k:Trajectories(root,split['train'],k) for k in ['edge','fork']};loaders={}
    for kind,ds in train.items():
        pos=ds.y.sum();neg=len(ds)-pos;w=np.where(ds.y>0,neg/max(pos,1)*(.5 if kind=='edge' else .333),1)
        sampler=WeightedRandomSampler(torch.from_numpy(w),max(len(ds),a.batch*100),replacement=True)
        loaders[kind]=DataLoader(ds,batch_size=a.batch,sampler=sampler,num_workers=3,persistent_workers=True,pin_memory=True,drop_last=True)
    iters={k:iter(v) for k,v in loaders.items()};config={'width':a.width,'layers':a.layers};model=TrajectoryNet(**config).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.03)
    history=[];best=-1.;start=time.time();print('TRAJECTORY_TRAIN_START',json.dumps({'config':config,'examples':{k:len(v) for k,v in train.items()},'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    for step in range(1,a.steps+1):
        if time.time()-start>a.seconds:break
        kind='fork' if step%2==0 else 'edge'
        try:x,g,y=next(iters[kind])
        except StopIteration:iters[kind]=iter(loaders[kind]);x,g,y=next(iters[kind])
        x,g,y=x.cuda().float(),g.cuda().float(),y.cuda();x,g=augment(x,g);model.train();opt.zero_grad(set_to_none=True)
        progress=max(step/a.steps,(time.time()-start)/a.seconds);lr=3e-4*(.15+.85*(1+np.cos(min(1,progress)*np.pi))/2)*min(1,step/100)
        for group in opt.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,g,kind);bce=F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none');pt=torch.where(y>.5,logits.sigmoid(),1-logits.sigmoid()).float();loss=(bce*(1-pt).clamp_min(.1)).mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if step%100==0:print('TRAJECTORY_STEP',step,kind,'loss',float(loss.detach()),'seconds',round(time.time()-start),flush=True)
        if step%1000==0:
            stats=validate(model,root,split['calibration'],a.batch);objective=stats['edge']['ap']+.45*stats['fork']['ap'];stats.update(step=step,seconds=round(time.time()-start),objective=objective);history.append(stats)
            ckpt={'config':config,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step,'validation':stats,'split':split,'training_arguments':vars(a)};atomic_save(ckpt,out/'last.pt')
            if objective>best:best=objective;atomic_save(ckpt,out/'best.pt')
            if step%5000==0:atomic_save(ckpt,out/('step_'+str(step)+'.pt'))
            (out/'history.json').write_text(json.dumps(history,indent=2));print('TRAJECTORY_VALIDATION',json.dumps(stats),flush=True)
    assert (out/'best.pt').exists();(out/'done.json').write_text(json.dumps({'step':step,'seconds':time.time()-start,'best_objective':best}));print('TRAJECTORY_TRAIN_DONE',step,best,flush=True)
