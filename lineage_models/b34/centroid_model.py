import torch
from torch import nn
from cell_event import CellEventNet
class CentroidNet(nn.Module):
    def __init__(self,base_config):
        super().__init__();self.base=CellEventNet(**base_config);self.head=nn.Sequential(nn.Linear(67,128),nn.SiLU(),nn.Linear(128,4));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias);self.head[-1].bias.data[-1]=1.8
    def forward(self,x,fraction):
        z=self.base.encode(x);raw=self.head(torch.cat([z,fraction.to(z.dtype)],1));return 4*torch.tanh(raw[:,:3]/4),torch.nn.functional.softplus(raw[:,3])
def load_centroid(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);model=CentroidNet(ck['config']['base_config']).cuda();model.load_state_dict(ck['model']);model.eval();return model,ck
