"""Permutation-equivariant candidate competition and joint daughter-pair selection."""
import torch
from torch import nn

class TrackSetNet(nn.Module):
    def __init__(self,input_dim=597,width=96,depth=2,structured=False,**kwargs):
        super().__init__();self.input_dim=input_dim;self.width=width;self.structured=structured
        self.project=nn.Sequential(nn.Linear(input_dim,width),nn.LayerNorm(width),nn.SiLU(),nn.Dropout(.15))
        layer=nn.TransformerEncoderLayer(width,4,width*3,.2,batch_first=True,norm_first=True,activation='gelu')
        self.competition=nn.TransformerEncoder(layer,depth,enable_nested_tensor=False)
        self.edge=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,1))
        self.count=nn.Sequential(nn.Linear(width*2,96),nn.SiLU(),nn.Dropout(.2),nn.Linear(96,3))
        self.pair=nn.Sequential(nn.Linear(width*4,96),nn.SiLU(),nn.Dropout(.2),nn.Linear(96,1))
    def forward(self,x,valid):
        x=torch.sign(x)*torch.log1p(x.abs());h=self.competition(self.project(x),src_key_padding_mask=~valid)
        mean=(h*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        maximum=h.masked_fill(~valid[:,:,None],-1e4).amax(1);parent=torch.cat([mean,maximum],1)
        count=self.count(parent);edge=self.edge(h).squeeze(-1).masked_fill(~valid,-1e4)
        a=h[:,:,None];b=h[:,None,:];shape=(len(h),h.shape[1],h.shape[1],self.width)
        pair=self.pair(torch.cat([(a+b).expand(shape)/2,(a-b).abs().expand(shape),(a*b).expand(shape),mean[:,None,None].expand(shape)],-1)).squeeze(-1)
        pairs=valid[:,:,None]&valid[:,None,:]&torch.ones((h.shape[1],h.shape[1]),device=h.device,dtype=torch.bool).triu(1)
        pair=pair.masked_fill(~pairs,-1e4)
        return edge,count,pair

def mask_incumbent(x):
    x=x.clone();x[...,44]=.5;x[...,45]=0.
    return x

def event_probabilities(edge,count,pair,valid,structured=False):
    counts=count.float().softmax(-1);p=edge.float().sigmoid()
    # A complete event probability: division existence times the selected pair.
    pair_probability=pair.float().flatten(1).softmax(-1).reshape_as(pair)*counts[:,2,None,None]
    pairs=valid[:,:,None]&valid[:,None,:]&torch.ones((valid.shape[1],valid.shape[1]),device=valid.device,dtype=torch.bool).triu(1)
    pair_probability=pair_probability*pairs
    if structured:
        p=counts[:,1,None]*edge.float().softmax(-1)+pair_probability.sum(1)+pair_probability.sum(2)
    return p*valid,pair_probability
