"""Frozen public temporal U-Net features with newly trained lineage-event heads.

The backbone class and initial weights are supplied by the separately attached
Pilkwang support package; see its original source and attribution.
"""
from pathlib import Path
import os,sys,json
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel,SDPBackend
from cell_event import CellEventNet

class PretrainedEventNet(CellEventNet):
    def __init__(self,seed=20260928,context=5,width=32,architecture='pretrained_unet',initialize=False):
        super().__init__(seed=seed,context=context,width=width)
        del self.encoder
        repo=Path(os.environ.get('BIOHUB_BASE_REPO','/workspace/biohub/baseline_setup/tracking_repo'))
        sys.path.insert(0,str(repo/'src'))
        from biohub_tracking.models.temporal_unet import TemporalUNet3D
        self.teacher=TemporalUNet3D(in_channels=1,out_channels=32,layers=[32,64,128],gradient_checkpointing=False)
        self.projection=nn.Sequential(nn.Linear(context*64,192),nn.SiLU(),nn.Dropout(.15),nn.Linear(192,64),nn.LayerNorm(64))
        if initialize:
            path=repo/'weights/unet_transformer/split_0/edge_predictor_best.pth'
            state=torch.load(path,map_location='cpu',weights_only=True)
            selected={k.removeprefix('unet.'):v for k,v in state.items() if k.startswith('unet.')}
            assert selected,'Missing pretrained U-Net keys'
            self.teacher.load_state_dict(selected,strict=True)
        self.teacher.requires_grad_(False);self.teacher.eval()
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def encode(self,x):
        if x.shape[1]!=self.context:
            assert x.shape[1]>self.context
            offset=(x.shape[1]-self.context)//2;x=x[:,offset:offset+self.context]
        z,y,w=x.shape[-3:]
        nucleus=x[:,:,z//2-1:z//2+2,y//2-3:y//2+4,w//2-3:w//2+4]
        photo=torch.cat([nucleus.mean((-3,-2,-1)),nucleus.std((-3,-2,-1))],dim=1)
        photo=self.photometry(torch.log1p(photo.clamp_min(0)))
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            features=self.teacher(x[...,::2,::2].unsqueeze(2))
            z,y,w=features.shape[-3:]
            local=features[:,:,:,z//2-1:z//2+2,y//2-1:y//2+2,w//2-1:w//2+2].mean((-3,-2,-1))
            broad=features.mean((-3,-2,-1))
            vector=torch.cat([local,broad],dim=2).flatten(1)
        return self.projection(vector)+.2*photo
