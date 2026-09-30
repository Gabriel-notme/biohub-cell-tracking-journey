"""Learned five-frame cell appearance and lineage-event models.

All image preparation and model fitting are executed on the cloud GPU pod.
Coordinates passed to geometry functions are physical micrometres.
"""
from pathlib import Path
from collections import OrderedDict
import json, math, os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCALE=np.array([1.625,.40625,.40625],np.float32)
PATCH=(12,24,24)
STRIDE=np.array([1,2,2])

def chain(start,links,pos,limit=4):
    result=[pos[start]]; cur=start
    for _ in range(limit-1):
        nxt=links.get(cur)
        if isinstance(nxt,list): nxt=nxt[0] if len(nxt)==1 else None
        if nxt is None or nxt not in pos: break
        result.append(pos[nxt]);cur=nxt
    return np.asarray(result,np.float32)

def edge_geometry(history,future):
    p=history[0];q=future[0]
    v=(p-history[-1])/max(1,len(history)-1)
    w=(future[-1]-q)/max(1,len(future)-1)
    d=q-p
    return np.r_[d/10,np.linalg.norm(d)/10,v/10,(d-v)/10,w/10,
                 np.linalg.norm(w-v)/10,len(history)>1,len(future)>1].astype(np.float32)

def fork_geometry(history,a,b):
    p=history[0];v=(p-history[-1])/max(1,len(history)-1)
    da=a[0]-p;db=b[0]-p
    dista=np.linalg.norm(da);distb=np.linalg.norm(db)
    result=[min(dista,distb)/10,max(dista,distb)/10,np.linalg.norm(a[0]-b[0])/10,
            np.dot(da,db)/(dista*distb+1e-4),np.linalg.norm((a[0]+b[0])/2-p-v)/10,
            abs(dista-distb)/10,np.linalg.norm(v)/10]
    for k in range(4):
        qa=a[min(k,len(a)-1)];qb=b[min(k,len(b)-1)]
        result.extend([np.linalg.norm(qa-qb)/10,np.linalg.norm((qa+qb)/2-p-(k+1)*v)/10,
                       min(np.linalg.norm(qa-p),np.linalg.norm(qb-p))/10,
                       max(np.linalg.norm(qa-p),np.linalg.norm(qb-p))/10])
    va=(a[-1]-a[0])/max(1,len(a)-1);vb=(b[-1]-b[0])/max(1,len(b)-1)
    result.extend([np.linalg.norm(va-vb)/10,np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-4),
                   len(history)/4,min(len(a),len(b))/4,max(len(a),len(b))/4])
    return np.asarray(result,np.float32)

class Movie:
    patch_shape=PATCH
    def __init__(self,path,context=5):
        import zarr
        self.group=zarr.open_group(str(path),mode='r');self.array=self.group['0']
        self.shape=self.array.shape;self.cache=OrderedDict();self.context=context
        qs=self.group.attrs.get('image_statistics',{}).get('quantiles',{})
        self.low=float(qs.get('0.001',0));self.high=float(qs.get('0.999',0))
        if self.high<=self.low:
            values=np.asarray(self.array[0])[::2,::4,::4]
            self.low,self.high=np.percentile(values,[.1,99.9])
    def frame(self,t):
        t=int(np.clip(t,0,self.shape[0]-1))
        if t not in self.cache:
            self.cache[t]=np.asarray(self.array[t,::1,::2,::2],np.float32)
            while len(self.cache)>max(7,self.context+2):self.cache.popitem(last=False)
        self.cache.move_to_end(t)
        return self.cache[t]
    def patch(self,t,coord):
        center=np.rint(np.asarray(coord)/STRIDE).astype(int)
        lo=center-np.array(PATCH)//2
        slices=[np.clip(np.arange(l,l+n),0,s-1) for l,n,s in zip(lo,PATCH,self.frame(t).shape)]
        patches=np.stack([self.frame(t+dt)[np.ix_(*slices)] for dt in range(-(self.context//2),self.context//2+1)])
        patches=(patches-self.low)/(self.high-self.low+1e-6)
        return np.clip(patches,0,3).astype(np.float16)
    def patches(self,times,coords):
        """Batch the same integer gather as patch(), preserving clipping and dtype."""
        times=np.asarray(times,dtype=np.int64);coords=np.asarray(coords)
        shape=tuple(self.patch_shape);result=np.empty((len(times),self.context,*shape),np.float16)
        center=np.rint(coords/STRIDE).astype(int);low=center-np.asarray(shape)//2
        for t in sorted(set(times.tolist())):
            indices=np.flatnonzero(times==t);frame_shape=self.frame(t).shape
            slices=[np.clip(low[indices,j,None]+np.arange(n)[None],0,s-1) for j,(n,s) in enumerate(zip(shape,frame_shape))]
            for channel,dt in enumerate(range(-(self.context//2),self.context//2+1)):
                values=self.frame(t+dt)[slices[0][:,:,None,None],slices[1][:,None,:,None],slices[2][:,None,None,:]]
                result[indices,channel]=np.clip((values-self.low)/(self.high-self.low+1e-6),0,3).astype(np.float16)
        return result

class Residual(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv3d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c),nn.SiLU(),nn.Conv3d(c,c,3,padding=1,bias=False),nn.GroupNorm(8,c))
    def forward(self,x):return F.silu(x+self.net(x))

class CenterContextPool(nn.Module):
    def forward(self,x):
        z,y,w=x.shape[-3:]
        center=x[:,:,z//2:z//2+1,max(0,y//2-1):y//2+1,max(0,w//2-1):w//2+1].mean((-3,-2,-1))
        return torch.cat([center,x.mean((-3,-2,-1))],dim=1)

class CellEventNet(nn.Module):
    def __init__(self,seed=20260921,context=5,width=24,normalization_context=None):
        super().__init__();self.context=context;self.normalization_context=normalization_context
        self.encoder=nn.Sequential(nn.Conv3d(context,width,3,padding=1,bias=False),nn.GroupNorm(8,width),nn.SiLU(),
            Residual(width),nn.Conv3d(width,width*2,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,width*2),nn.SiLU(),
            Residual(width*2),nn.Conv3d(width*2,width*4,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,width*4),nn.SiLU(),
            CenterContextPool(),nn.Linear(width*8,64),nn.LayerNorm(64))
        self.edge=nn.Sequential(nn.Linear(64*4+16,192),nn.SiLU(),nn.Dropout(.15),nn.Linear(192,96),nn.SiLU(),nn.Linear(96,1))
        self.fork=nn.Sequential(nn.Linear(64*4+28,192),nn.SiLU(),nn.Dropout(.25),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))
        self.phase=nn.Linear(64,1)
        self.photometry=nn.Sequential(nn.Linear(context*2,64),nn.SiLU(),nn.Linear(64,64))
    def encode(self,x):
        if x.shape[1]!=self.context:
            assert x.shape[1]>self.context
            offset=(x.shape[1]-self.context)//2;x=x[:,offset:offset+self.context]
        z,y,w=x.shape[-3:]
        nucleus=x[:,:,z//2-1:z//2+2,y//2-3:y//2+4,w//2-3:w//2+4]
        photo=torch.cat([nucleus.mean((-3,-2,-1)),nucleus.std((-3,-2,-1))],dim=1)
        photo=self.photometry(torch.log1p(photo.clamp_min(0)))
        # Shared global normalization preserves temporal brightness changes.
        normalization=x
        if self.normalization_context is not None:
            offset=(self.context-self.normalization_context)//2
            normalization=x[:,offset:offset+self.normalization_context]
        mean=normalization.mean((1,2,3,4),keepdim=True);sd=normalization.std((1,2,3,4),keepdim=True).clamp_min(.03)
        return self.encoder(((x-mean)/sd).clamp(-5,10))+.2*photo
    def edge_logits(self,a,b,g):
        return self.edge(torch.cat([a,b,(a-b).abs(),a*b,g],-1)).squeeze(-1)
    def fork_logits(self,p,a,b,g):
        return self.fork(torch.cat([p,(a+b)/2,(a-b).abs(),a*b,g],-1)).squeeze(-1)

def load_event_model(path,device='cuda'):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if ckpt['config'].get('architecture')=='pretrained_unet':
        from pretrained_event import PretrainedEventNet
        model=PretrainedEventNet(**ckpt['config']).to(device)
    else:model=CellEventNet(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model']);model.eval()
    return model,ckpt
