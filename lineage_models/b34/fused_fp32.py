"""Standard inference-only Conv3d/BatchNorm folding; full FP32 arithmetic."""
import torch

def install_fused_fp32(model):
    assert not model.training
    count=0
    for module in list(model.modules()):
        if isinstance(module,torch.nn.Sequential):
            for i in range(len(module)-1):
                if isinstance(module[i],torch.nn.Conv3d) and isinstance(module[i+1],torch.nn.BatchNorm3d):
                    module[i]=torch.nn.utils.fuse_conv_bn_eval(module[i],module[i+1]);module[i+1]=torch.nn.Identity();count+=1
    assert count>0,'Expected convolution/BatchNorm pairs'
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    print('FP32_BATCHNORM_FUSED',count,flush=True)
    return model
