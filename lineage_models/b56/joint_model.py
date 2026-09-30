"""Parent-conditioned spatial-temporal lineage network with dense offspring supervision."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cell_event import Movie,Residual,SCALE

SHAPE=(16,40,40)
SPACING=np.array([1.625,.8125,.8125],np.float32)

class JointMovie(Movie):
    patch_shape=SHAPE
    def __init__(self,path,context=7):super().__init__(path,context)
    def patch(self,t,coord):
        center=np.rint(np.asarray(coord)/[1,2,2]).astype(int)
        low=center-np.array(SHAPE)//2
        slices=[np.clip(np.arange(l,l+n),0,s-1) for l,n,s in zip(low,SHAPE,self.frame(t).shape)]
        x=np.stack([self.frame(t+dt)[np.ix_(*slices)] for dt in range(-(self.context//2),self.context//2+1)])
        return np.clip((x-self.low)/(self.high-self.low+1e-6),0,3).astype(np.float16)

def conv(a,b,stride=1):
    return nn.Sequential(nn.Conv3d(a,b,3,stride=stride,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU(),Residual(b))

class JointLineageNet(nn.Module):
    def __init__(self,context=7,width=24,architecture='joint',parent_condition=False,robust_pool=False,**kwargs):
        super().__init__();self.context=context;self.architecture=architecture;self.parent_condition=parent_condition;self.robust_pool=robust_pool
        if architecture=='temporal_attention':
            self.frame_stem=nn.Sequential(nn.Conv3d(1,8,3,padding=1),nn.SiLU(),nn.Conv3d(8,8,3,padding=1),nn.SiLU())
            self.time_gate=nn.Sequential(nn.Linear(8,24),nn.SiLU(),nn.Linear(24,1))
            self.stem=conv(context*8+context+int(parent_condition),width)
        else:self.stem=conv(context+int(parent_condition),width)
        self.down1=conv(width,width*2,2);self.down2=conv(width*2,width*4,2)
        self.up1=conv(width*6,width*2);self.up0=conv(width*3,width)
        self.heatmap=nn.Conv3d(width,1,1)
        self.phase=nn.Sequential(nn.Linear(width*8,96),nn.SiLU(),nn.Dropout(.2),nn.Linear(96,1))
        # p,a,b: multiscale spatial features + temporal photometry + dense offspring response.
        self.dim=width*7+context+1
        self.edge=nn.Sequential(nn.Linear(self.dim*4+16,256),nn.SiLU(),nn.Dropout(.15),nn.Linear(256,96),nn.SiLU(),nn.Linear(96,1))
        self.fork=nn.Sequential(nn.Linear(self.dim*4+28,256),nn.SiLU(),nn.Dropout(.3),nn.Linear(256,96),nn.SiLU(),nn.Linear(96,1))
    def encode(self,x,parent_relative=None):
        if x.shape[1]!=self.context:
            off=(x.shape[1]-self.context)//2;x=x[:,off:off+self.context]
        photo=x
        mean=x.mean((1,2,3,4),keepdim=True);sd=x.std((1,2,3,4),keepdim=True).clamp_min(.03)
        x=((x-mean)/sd).clamp(-5,10)
        if self.architecture=='temporal_attention':
            b,t,z,y,w=x.shape;f=self.frame_stem(x.reshape(b*t,1,z,y,w)).reshape(b,t,8,z,y,w)
            gate=self.time_gate(f.mean((-3,-2,-1))).softmax(1)*t
            x=torch.cat([x,(f*gate[...,None,None,None]).flatten(1,2)],1)
        if self.parent_condition:
            axes=[(torch.arange(n,device=x.device,dtype=x.dtype)-n//2)*float(s) for n,s in zip(SHAPE,SPACING)]
            grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)
            parent=parent_relative if parent_relative is not None else x.new_zeros((len(x),3))
            mask=torch.exp(-.5*((grid[None]-parent[:,None,None,None,:])/2.).square().sum(-1)).unsqueeze(1)
            x=torch.cat([x,mask.to(x.dtype)],1)
        a=self.stem(x);b=self.down1(a);c=self.down2(b)
        d=self.up1(torch.cat([F.interpolate(c,size=b.shape[-3:],mode='trilinear',align_corners=True),b],1))
        e=self.up0(torch.cat([F.interpolate(d,size=a.shape[-3:],mode='trilinear',align_corners=True),a],1))
        heat=self.heatmap(e)
        cz,cy,cx=c.shape[-3:]
        phase=self.phase(torch.cat([c.mean((-3,-2,-1)),c[:,:,cz//2,cy//2,cx//2]],1)).squeeze(-1)
        return e,b,c,photo,heat,phase
    def pool(self,encoded,relative_um):
        # Exact align_corners mapping for the original parent-centered crop.
        e,b,c,photo,heat,_=encoded
        if self.robust_pool:
            if getattr(self,'fast_photo_pool',False):
                photo=F.avg_pool3d(photo,kernel_size=(3,1,1),stride=1,padding=(1,0,0))
                photo=F.avg_pool3d(photo,kernel_size=(1,7,1),stride=1,padding=(0,3,0))
                photo=F.avg_pool3d(photo,kernel_size=(1,1,7),stride=1,padding=(0,0,3))
            else:photo=F.avg_pool3d(photo,kernel_size=(3,7,7),stride=1,padding=(1,3,3))
            heat=F.max_pool3d(heat,kernel_size=(3,5,5),stride=1,padding=(1,2,2))
        shape=relative_um.new_tensor(SHAPE);spacing=relative_um.new_tensor(SPACING)
        xyz=relative_um/spacing+shape.floor_divide(2)
        grid=(xyz/(shape-1)*2-1).flip(-1).reshape(len(relative_um),-1,1,1,3)
        def take(a):return F.grid_sample(a,grid.to(a.dtype),mode='bilinear',padding_mode='border',align_corners=True)[:,:,:,0,0].transpose(1,2)
        return torch.cat([take(e),take(b),take(c),torch.log1p(take(photo).clamp_min(0)),take(heat).sigmoid()],-1)
    def classify(self,encoded,relative_um,g,kind):
        features=self.pool(encoded,relative_um)
        return self.classify_features(features,g,kind)
    def classify_features(self,features,g,kind):
        p,a=features[:,0],features[:,1]
        if kind=='edge':return self.edge(torch.cat([p,a,(p-a).abs(),p*a,g],-1)).squeeze(-1)
        b=features[:,2]
        return self.fork(torch.cat([p,(a+b)/2,(a-b).abs(),a*b,g],-1)).squeeze(-1)
    def forward(self,x,relative_um,g,kind):
        encoded=self.encode(x,relative_um[:,0])
        return self.classify(encoded,relative_um,g,kind),encoded[-2],encoded[-1]

def load_joint(path,device='cuda'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=JointLineageNet(**checkpoint['config']).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint
