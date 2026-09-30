"""Raw-image tracklet encoder with long temporal context and symmetric event heads."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cell_event import Movie,SCALE

STEPS=10
PATCH=(8,16,16)

class TrackMovie(Movie):
    def __init__(self,path,channels=1):super().__init__(path,context=channels);self.channels=channels
    def crops(self,t,coords):
        frame=self.frame(int(t));centers=np.rint(np.asarray(coords)/[1,2,2]).astype(int)
        axes=[np.clip(centers[:,j,None]+np.arange(n)-n//2,0,frame.shape[j]-1) for j,n in enumerate(PATCH)]
        take=lambda f:f[axes[0][:,:,None,None],axes[1][:,None,:,None],axes[2][:,None,None,:]]
        x=take(frame) if self.channels==1 else np.stack([take(self.frame(t+d)) for d in range(-(self.channels//2),self.channels//2+1)],1)
        return np.clip((x-self.low)/(self.high-self.low+1e-6),0,4).astype(np.float16)

def walk(n,links,limit=STEPS):
    chain=[n]
    for _ in range(limit-1):
        p=links.get(chain[-1])
        if isinstance(p,list):p=p[0] if len(p)==1 else None
        if p is None:break
        chain.append(p)
    return chain

def sequences(ids,nodes,out,prev):
    lookup={n:i for i,n in enumerate(ids)};index=[];motion=[]
    for links in [prev,out]:
        ii=np.zeros((len(ids),STEPS),np.int64);xx=np.zeros((len(ids),STEPS,4),np.float32)
        for k,n in enumerate(ids):
            ch=walk(n,links);ch=[q for q in ch if q in lookup]
            p=np.asarray([[nodes[q][v] for v in ['z','y','x']] for q in ch],np.float32)*SCALE
            pp=np.stack([p[min(j,len(p)-1)] for j in range(STEPS)])
            ii[k]=[lookup[ch[min(j,len(ch)-1)]] for j in range(STEPS)]
            xx[k,:,:3]=(pp-pp[:1])/10;xx[k,:,3]=(np.arange(STEPS)<len(ch))
        index.append(ii);motion.append(xx)
    return np.stack(index),np.stack(motion)

class TrackVideoNet(nn.Module):
    def __init__(self,width=64,channels=1,edge_dim=16,fork_dim=28,**kwargs):
        super().__init__();self.width=width;self.channels=channels;self.edge_dim=edge_dim;self.fork_dim=fork_dim
        self.frame=nn.Sequential(nn.Conv3d(channels,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv3d(16,24,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,24),nn.SiLU(),
            nn.Conv3d(24,32,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,32),nn.SiLU())
        self.frame_proj=nn.Sequential(nn.Linear(96+4*channels,width),nn.LayerNorm(width),nn.SiLU())
        self.motion=nn.Linear(4,width)
        self.time=nn.Parameter(torch.randn(1,STEPS,width)*.02)
        layer=nn.TransformerEncoderLayer(width,4,width*3,.15,batch_first=True,norm_first=True,activation='gelu')
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.track=nn.Sequential(nn.Linear(width*3,width),nn.LayerNorm(width),nn.SiLU())
        self.edge_geometry=nn.Identity() if edge_dim==16 else nn.Sequential(nn.Linear(edge_dim,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2))
        self.fork_geometry=nn.Identity() if fork_dim==28 else nn.Sequential(nn.Linear(fork_dim,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2))
        self.edge=nn.Sequential(nn.Linear(width*4+(16 if edge_dim==16 else 64),192),nn.SiLU(),nn.Dropout(.2),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))
        self.fork=nn.Sequential(nn.Linear(width*4+(28 if fork_dim==28 else 64),192),nn.SiLU(),nn.Dropout(.3),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))
        self.phase=nn.Linear(width,1)
    def encode_frame(self,x):
        x=x.float();mean=x.mean((-3,-2,-1),keepdim=True);std=x.std((-3,-2,-1),keepdim=True).clamp_min(.03)
        h=self.frame(((x-mean)/std).clamp(-4,8))
        photo=torch.cat([mean.flatten(1),std.flatten(1),x.amax((-3,-2,-1)),x[:,:,2:6,4:12,4:12].mean((-3,-2,-1))],1)
        h=torch.cat([h.mean((-3,-2,-1)),h.amax((-3,-2,-1)),h[:,:,h.shape[2]//2,h.shape[3]//2,h.shape[4]//2],torch.log1p(photo)],1)
        return self.frame_proj(h)
    def encode_track(self,z,m):
        valid=m[:,:,3]>.5
        h=self.temporal(z+self.motion(m)+self.time,src_key_padding_mask=~valid)
        mean=(h*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        last=h[torch.arange(len(h),device=h.device),valid.sum(1).clamp_min(1)-1]
        return self.track(torch.cat([h[:,0],mean,last],1))
    def classify(self,z,g,kind):
        if (self.edge_dim if kind=='edge' else self.fork_dim) not in (16,28):g=torch.sign(g)*torch.log1p(g.abs())
        g=(self.edge_geometry if kind=='edge' else self.fork_geometry)(g)
        p,a=z[:,0],z[:,1]
        f=torch.cat([p,a,(p-a).abs(),p*a,g],1) if kind=='edge' else torch.cat([p,(a+z[:,2])/2,(a-z[:,2]).abs(),a*z[:,2],g],1)
        return (self.edge if kind=='edge' else self.fork)(f).squeeze(1)
    def forward(self,x,m,g,kind):
        b,k,t=x.shape[:3]
        z=self.encode_frame(x.reshape(-1,self.channels,*PATCH)).reshape(b*k,t,self.width)
        z=self.encode_track(z,m.reshape(b*k,t,4)).reshape(b,k,self.width)
        return self.classify(z,g,kind),self.phase(z[:,0]).squeeze(1)

def load_track(path,device='cuda'):
    torch.backends.mha.set_fastpath_enabled(False)
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    cls=TrackVideoNet
    if ckpt['config'].get('architecture')=='context_track':
        from context_track import ContextTrackNet
        cls=ContextTrackNet
    model=cls(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model']);model.eval();return model,ckpt
