"""Trainable seven-frame lineage expert initialized by the established B1 model."""
import torch
from torch import nn
from cell_event import CellEventNet

class TransferLineageNet(nn.Module):
    def __init__(self,base_config=None,width=96,layers=2,**kwargs):
        super().__init__();self.base=CellEventNet(**(base_config or {'context':3,'width':24}))
        for p in self.base.parameters():p.requires_grad_(False)
        self.proj=nn.Sequential(nn.Linear(64,width),nn.LayerNorm(width),nn.SiLU());self.time=nn.Parameter(torch.randn(1,3,width)*.02)
        block=nn.TransformerEncoderLayer(width,4,width*3,.2,batch_first=True,norm_first=True,activation='gelu')
        self.transformer=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.summary=nn.Sequential(nn.Linear(width*3,96),nn.LayerNorm(96),nn.SiLU())
        self.edge=nn.Sequential(nn.Linear(96*4+16,192),nn.SiLU(),nn.Dropout(.25),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))
        self.fork=nn.Sequential(nn.Linear(96*4+28,192),nn.SiLU(),nn.Dropout(.35),nn.Linear(192,64),nn.SiLU(),nn.Linear(64,1))
        for head in [self.edge,self.fork]:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def encode(self,x):
        z=self.transformer(self.proj(x)+self.time);return self.summary(z.flatten(1))
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def forward(self,x,g,kind,return_base=False):
        shape=x.shape;z=self.encode(x.flatten(0,1)).reshape(shape[0],shape[1],-1)
        return self.classify_features(z,x[:,:,1],g,kind,return_base)
    def classify_features(self,z,central,g,kind,return_base=False):
        p,a=z[:,0],z[:,1]
        if kind=='edge':
            base=self.base.edge_logits(central[:,0],central[:,1],g);delta=self.edge(torch.cat([p,a,(p-a).abs(),p*a,g],-1)).squeeze(-1)
        else:
            b=z[:,2];base=self.base.fork_logits(central[:,0],central[:,1],central[:,2],g);delta=self.fork(torch.cat([p,(a+b)/2,(a-b).abs(),a*b,g],-1)).squeeze(-1)
        pred=base+delta
        return (pred,base,delta) if return_base else pred

def load_transfer(path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=False);m=TransferLineageNet(**ck['config']).to(device);m.load_state_dict(ck['model']);m.eval();return m,ck
