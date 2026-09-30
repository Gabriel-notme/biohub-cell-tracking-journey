"""Track-following temporal appearance expert, complementary to fixed-view 3D CNNs."""
import numpy as np,torch
from torch import nn
from cell_event import Movie,SCALE

STEPS=8
class AppearanceMovie(Movie):
    def __init__(self,path):super().__init__(path,context=3)
    def descriptors(self,items):
        result={};bytime={}
        for key,(t,c) in items.items():bytime.setdefault(int(t),[]).append((key,np.asarray(c)))
        for t,rows in sorted(bytime.items()):
            frame=self.frame(t);shape=np.asarray(frame.shape)
            for start in range(0,len(rows),512):
                chunk=rows[start:start+512];centers=np.rint(np.stack([c for k,c in chunk])/[1,2,2]).astype(int)
                zz=np.clip(centers[:,0,None]+np.arange(-3,4),0,shape[0]-1);yy=np.clip(centers[:,1,None]+np.arange(-6,7),0,shape[1]-1);xx=np.clip(centers[:,2,None]+np.arange(-6,7),0,shape[2]-1)
                x=frame[zz[:,:,None,None],yy[:,None,:,None],xx[:,None,None,:]]
                x=np.clip((x-self.low)/(self.high-self.low+1e-6),0,3);flat=x.reshape(len(x),-1);central=x[:,2:5,3:10,3:10].reshape(len(x),-1)
                outerq=np.quantile(flat,[.1,.5,.9,.99],axis=1).T;innerq=np.quantile(central,[.1,.5,.9,.99],axis=1).T
                weight=np.maximum(0,x-outerq[:,1,None,None,None]);mass=weight.sum((1,2,3))+1e-6
                axes=[np.arange(-3,4)*1.625,np.arange(-6,7)*.8125,np.arange(-6,7)*.8125];grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
                center=(weight[...,None]*grid[None]).sum((1,2,3))/mass[:,None]
                var=(weight[...,None]*(grid[None]-center[:,None,None,None,:])**2).sum((1,2,3))/mass[:,None]
                sharp=np.abs(np.diff(x,axis=2)).mean((1,2,3))+np.abs(np.diff(x,axis=3)).mean((1,2,3))
                f=np.c_[central.mean(1),central.std(1),central.max(1),innerq,flat.mean(1),flat.std(1),flat.max(1),outerq,
                    mass/100,np.linalg.norm(center,axis=1)/5,np.sqrt(var.sum(1))/5,var[:,0]/25,(var[:,1]+var[:,2])/50,np.abs(var[:,1]-var[:,2])/25,
                    (central.mean(1)-outerq[:,1])/(outerq[:,3]-outerq[:,1]+.01),sharp,
                    (flat>outerq[:,3,None]*.5).mean(1),(flat>outerq[:,3,None]*.8).mean(1)]
                assert f.shape[1]==24,f.shape
                f=np.nan_to_num(f).astype(np.float32)
                for (key,c),v in zip(chunk,f):result[key]=v
        return result

def track_nodes(start,links,nodes,limit=STEPS):
    ids=[start];cur=start
    for _ in range(limit-1):
        nxt=links.get(cur)
        if isinstance(nxt,list):nxt=nxt[0] if len(nxt)==1 else None
        if nxt is None or nxt not in nodes:break
        ids.append(nxt);cur=nxt
    return ids

def sequence_features(points,descriptors,keys):
    # Points are current-to-past for parent, current-to-future for daughters.
    n=len(points);p=np.asarray(points,np.float32);motion=np.zeros((STEPS,5),np.float32)
    position=np.stack([p[min(i,n-1)] for i in range(STEPS)])
    delta=np.diff(position,axis=0,prepend=position[:1]);acc=np.diff(delta,axis=0,prepend=delta[:1])
    motion[:,0]=np.linalg.norm(position-position[:1],axis=1)/10;motion[:,1]=np.linalg.norm(delta,axis=1)/10;motion[:,2]=np.linalg.norm(acc,axis=1)/10
    motion[:,3]=(np.arange(STEPS)<n).astype(np.float32);motion[:,4]=min(n,STEPS)/STEPS
    x=np.stack([descriptors[keys[min(i,n-1)]] for i in range(STEPS)])
    return np.c_[x,motion].astype(np.float32)

class TrajectoryNet(nn.Module):
    def __init__(self,width=96,layers=2,**kwargs):
        super().__init__();self.projection=nn.Sequential(nn.Linear(29,width),nn.LayerNorm(width),nn.SiLU())
        self.time=nn.Parameter(torch.randn(1,STEPS,width)*.02)
        block=nn.TransformerEncoderLayer(width,4,width*3,.15,batch_first=True,norm_first=True,activation='gelu')
        self.temporal=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.summary=nn.Sequential(nn.Linear(width*3,96),nn.LayerNorm(96),nn.SiLU())
        self.edge=nn.Sequential(nn.Linear(96*4+16,192),nn.SiLU(),nn.Dropout(.2),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))
        self.fork=nn.Sequential(nn.Linear(96*4+28,192),nn.SiLU(),nn.Dropout(.3),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))
    def encode(self,x):
        valid=x[:,:,27]>.5
        y=x.clone();y[:,:,:24]=torch.sign(y[:,:,:24])*torch.log1p(y[:,:,:24].abs())
        h=self.temporal(self.projection(y)+self.time,src_key_padding_mask=~valid)
        mean=(h*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None];last=h[torch.arange(len(h),device=h.device),valid.sum(1).clamp_min(1)-1]
        return self.summary(torch.cat([h[:,0],mean,last],-1))
    def forward(self,x,g,kind):
        z=self.encode(x.flatten(0,1)).reshape(len(x),x.shape[1],-1)
        return self.classify_features(z,g,kind)
    def classify_features(self,z,g,kind):
        p,a=z[:,0],z[:,1]
        if kind=='edge':return self.edge(torch.cat([p,a,(p-a).abs(),p*a,g],-1)).squeeze(-1)
        b=z[:,2];return self.fork(torch.cat([p,(a+b)/2,(a-b).abs(),a*b,g],-1)).squeeze(-1)

def load_trajectory(path,device='cuda'):
    ckpt=torch.load(path,map_location='cpu',weights_only=False);model=TrajectoryNet(**ckpt['config']).to(device);model.load_state_dict(ckpt['model']);model.eval();return model,ckpt
