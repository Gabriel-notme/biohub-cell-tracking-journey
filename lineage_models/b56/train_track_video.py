from pathlib import Path
import os,json,time,argparse,random,hashlib
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from track_video import TrackVideoNet,STEPS,PATCH

class Events(Dataset):
    def __init__(self,root,names,kind):
        self.root=Path(root);self.names=names;self.kind=kind;self.meta=[];self.crops={};rows=[];self.geom=[];importance=[]
        for mi,name in enumerate(names):
            with np.load(self.root/(name+'.npz')) as d:
                self.meta.append({k:d[k] for k in ['seq','motion','phase']});r=d[kind]
                rows.append(np.c_[np.full(len(r),mi),r]);self.geom.append(d[kind+'_geom']);importance.append(d[kind+'_weight'] if kind+'_weight' in d else np.ones(len(r),np.float32))
        self.rows=np.concatenate(rows).astype(np.int64);self.geom=np.concatenate(self.geom);self.importance=np.concatenate(importance)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        mi,*ns,y=self.rows[i];d=self.meta[mi]
        if mi not in self.crops:self.crops[mi]=np.load(self.root/(self.names[mi]+'_crops.npy'),mmap_mode='r')
        seq=[d['seq'][int(j>0),n] for j,n in enumerate(ns)]
        x=self.crops[mi][np.stack(seq)].astype(np.float32)
        m=np.stack([d['motion'][int(j>0),n] for j,n in enumerate(ns)])
        return x,m,self.geom[i].copy(),np.float32(y),d['phase'][ns[0]]

class EventPairs(Dataset):
    def __init__(self,data):
        self.data=data;positive={};negative={}
        for i,row in enumerate(data.rows):(positive if row[-1] else negative).setdefault(tuple(row[:2]),[]).append(i)
        self.groups=[(p,negative[k]) for k,p in positive.items() if k in negative]
    def __len__(self):return max(1024,len(self.groups)*16)
    def __getitem__(self,i):
        p,n=self.groups[i%len(self.groups)];a=self.data[random.choice(p)]
        ni=random.choice(n) if np.all(self.data.importance[n]==1) else random.choices(n,weights=self.data.importance[n],k=1)[0]
        b=self.data[ni]
        return tuple(np.stack([x,y]) for x,y in zip(a,b))

def augment(x,m,g,kind):
    for j,axis in enumerate([-3,-2,-1]):
        if random.random()<.5:
            x=x.flip(axis);m[...,j]*=-1
            if kind=='edge' and g.shape[1]==16:
                for k in [0,4,7,10]:g[:,k+j]*=-1
    if random.random()<.5:
        x=x.transpose(-1,-2);m[...,[1,2]]=m[...,[2,1]]
        if kind=='edge' and g.shape[1]==16:
            for k in [0,4,7,10]:g[:,[k+1,k+2]]=g[:,[k+2,k+1]]
    # Match detector-centering uncertainty without changing event identity.
    shifts=[random.randint(-1,1),random.randint(-2,2),random.randint(-2,2)]
    x=torch.roll(x,shifts,(-3,-2,-1))
    amp=torch.empty((len(x),)+(1,)*(x.ndim-1),device=x.device).uniform_(.4,1.8)
    x=(x*amp+torch.randn_like(x)*random.uniform(0,.04)).clamp_min(0)
    x=x*torch.empty((len(x),x.shape[1])+(1,)*(x.ndim-2),device=x.device).uniform_(.85,1.15)
    m[...,:3]+=torch.randn_like(m[...,:3])*.035
    if random.random()<.5:
        lengths=torch.randint(2,STEPS+1,(len(x),x.shape[1],1),device=x.device)
        valid=(torch.arange(STEPS,device=x.device)[None,None]<lengths)
        m[...,3]*=valid
    g+=torch.randn_like(g)*.025
    # Prevent a shortcut that merely repeats the inherited graph's decisions.
    if g.shape[1] in (338,530):
        selected=torch.rand(len(g),device=g.device)<.5
        offsets=[0] if kind=='edge' else [g.shape[1]-96,g.shape[1]-48]
        for offset in offsets:
            for column,value in [(44,.5),(45,0.),(46,1.),(47,0.)]:g[selected,offset+column]=value
    return x,m,g

@torch.inference_mode()
def evaluate(model,loaders):
    model.eval();result={}
    for kind,loader in loaders.items():
        ys=[];ps=[]
        for x,m,g,y,p in loader:
            with torch.autocast('cuda',dtype=torch.float16):logits,_=model(x.cuda(),m.cuda(),g.cuda(),kind)
            ys.extend(y.numpy());ps.extend(logits.float().sigmoid().cpu().numpy())
        y=np.asarray(ys);p=np.asarray(ps)
        result[kind]={'n':len(y),'positive':int(y.sum()),'ap':float(average_precision_score(y,p)),
            'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,'logloss':float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1]))}
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--seed',type=int,default=17);a.add_argument('--hold-embryo',default='');a.add_argument('--seconds',type=int,default=3600);a.add_argument('--steps',type=int,default=16000);a.add_argument('--batch',type=int,default=24);a.add_argument('--data',default='track_data');a.add_argument('--channels',type=int,default=1);a.add_argument('--initialize');a.add_argument('--ranking-weight',type=float,default=0);args=a.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    R=Path('/workspace/biohub');D=R/args.data;out=R/args.out;out.mkdir(exist_ok=True)
    assert not (out/'history.json').exists(),'Do not overwrite prior experiment'
    while not (D/'ready.json').exists():time.sleep(5)
    split=json.loads((D/'split.json').read_text());train=split['train'];valid=split['calibration']
    if args.hold_embryo:
        train=[n for n in train if not n.startswith(args.hold_embryo)]
        valid=[n for n in valid if n.startswith(args.hold_embryo)]
    datasets={k:Events(D,train,k) for k in ['edge','fork']};loaders={};validation={}
    for k,ds in datasets.items():
        y=ds.rows[:,-1];w=np.where(y,((len(y)-y.sum())/max(1,y.sum()))*(1 if k=='edge' else .5),1)
        sampler=WeightedRandomSampler(torch.from_numpy(w*ds.importance),max(len(y),args.batch*500),replacement=True)
        loaders[k]=DataLoader(ds,batch_size=args.batch,sampler=sampler,num_workers=4,persistent_workers=True,pin_memory=True,drop_last=True)
        validation[k]=DataLoader(Events(D,valid,k),batch_size=args.batch*2,num_workers=2,persistent_workers=True,pin_memory=True)
    dims={k:datasets[k].geom.shape[1] for k in ['edge','fork']}
    model=TrackVideoNet(channels=args.channels,edge_dim=dims['edge'],fork_dim=dims['fork']).cuda()
    if args.initialize:
        old=torch.load(R/args.initialize,map_location='cpu',weights_only=False)
        matching={k:v for k,v in old['model'].items() if k in model.state_dict() and v.shape==model.state_dict()[k].shape}
        incompatible=model.load_state_dict(matching,strict=False);print('INITIALIZATION',len(matching),'tensors;',incompatible,flush=True)
        with torch.no_grad():
            for kind in ['edge','fork']:
                key=kind+'.0.weight'
                if key in old['model'] and key not in matching:
                    getattr(model,kind)[0].weight[:,:model.width*4].copy_(old['model'][key][:,:model.width*4])
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.025);scaler=torch.amp.GradScaler('cuda')
    it={k:iter(v) for k,v in loaders.items()};history=[];best=-1;best_specialties={'edge':-1.,'fork':-1.};start=time.time();pairloaders={};pairit={}
    if args.ranking_weight:
        for k,ds in datasets.items():
            paired=EventPairs(ds)
            if paired.groups:
                pairloaders[k]=DataLoader(paired,batch_size=8,shuffle=True,num_workers=2,persistent_workers=True,pin_memory=True,drop_last=True);pairit[k]=iter(pairloaders[k])
    config={'width':64,'channels':args.channels,'edge_dim':dims['edge'],'fork_dim':dims['fork']};info={**vars(args),'train':train,'valid':valid,'initialization':args.initialize or 'random','config':config,'examples':{k:len(v) for k,v in datasets.items()}}
    info['sampling_importance']={k:{'maximum':float(v.importance.max()),'mean':float(v.importance.mean())} for k,v in datasets.items()}
    info['training_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if 'augmentation_source_movies' in split:
        info['augmentation_source_movies']=split['augmentation_source_movies']
        assert set(info['augmentation_source_movies'].values()).issubset(set(train))
        assert not set(info['augmentation_source_movies'].values())&set(valid+split['new_audit'])
    (out/'training_protocol.json').write_text(json.dumps(info,indent=2));print('TRACK_TRAIN_START',json.dumps(info),flush=True)
    for step in range(1,args.steps+1):
        if time.time()-start>args.seconds:break
        kind='fork' if step%2==0 else 'edge'
        try:batch=next(it[kind])
        except StopIteration:it[kind]=iter(loaders[kind]);batch=next(it[kind])
        pairs=0
        if kind in pairit:
            try:paired=next(pairit[kind])
            except StopIteration:pairit[kind]=iter(pairloaders[kind]);paired=next(pairit[kind])
            pairs=len(paired[0]);batch=[torch.cat([a,b.flatten(0,1)]) for a,b in zip(batch,paired)]
        x,m,g,y,phase=[v.cuda(non_blocking=True).float() for v in batch];x,m,g=augment(x,m,g,kind)
        progress=max(step/args.steps,(time.time()-start)/args.seconds);lr=3e-4*(.08+.92*(1+np.cos(min(1,progress)*np.pi))/2)*min(1,step/100)
        for group in opt.param_groups:group['lr']=lr
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits,ph=model(x,m,g,kind)
            loss=F.binary_cross_entropy_with_logits(logits.float(),y)+.08*F.binary_cross_entropy_with_logits(ph.float(),phase)
            if pairs:
                ranked=logits[-2*pairs:].float().reshape(-1,2);loss=loss+args.ranking_weight*F.softplus(1-ranked[:,0]+ranked[:,1]).mean()
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(opt);scaler.update()
        if step%100==0:print('TRAIN',step,kind,round(loss.item(),5),round(time.time()-start),flush=True)
        if step%1200==0:
            metrics=evaluate(model,validation);value=metrics['edge']['ap']+.5*metrics['fork']['ap'];row={'step':step,'seconds':time.time()-start,'metrics':metrics};history.append(row)
            ckpt={'model':model.state_dict(),'config':config,'step':step,'metrics':metrics,'protocol':info}
            torch.save(ckpt,out/'last.pt')
            if value>best:best=value;torch.save(ckpt,out/'best.pt')
            for specialty in ['edge','fork']:
                if metrics[specialty]['ap']>best_specialties[specialty]:
                    best_specialties[specialty]=metrics[specialty]['ap'];torch.save(ckpt,out/('best_'+specialty+'.pt'))
            (out/'history.json').write_text(json.dumps(history,indent=2));print('TRACK_VALIDATED',json.dumps(row),flush=True)
    if not history:
        metrics=evaluate(model,validation);history=[{'step':step,'metrics':metrics}]
        torch.save({'model':model.state_dict(),'config':config,'step':step,'metrics':metrics,'protocol':info},out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'ready.json').write_text(json.dumps({'seconds':time.time()-start,'steps':step}));print('TRACK_TRAIN_DONE',flush=True)
