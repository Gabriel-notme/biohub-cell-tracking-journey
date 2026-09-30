"""Batch the eight existing D4 encodes, preserving their FP32 accumulation order."""
import torch

def install_batched_d4(model,batch_size=2):
    assert batch_size in [2,4]
    original=model.encode;state={'index':0,'base':None,'pending':[]}
    def transform(x,i):
        if i==0:return x
        if i<=3:return x.flip([(-1,),(-2,),(-2,-1)][i-1])
        if i<=5:return torch.rot90(x,[1,3][i-4],(-2,-1))
        if i==6:return x.transpose(-1,-2)
        return torch.rot90(x,1,(-2,-1)).transpose(-1,-2)
    def encode(imgs):
        assert imgs.shape[0]==1 and imgs.shape[-1]==imgs.shape[-2]
        i=state['index']
        if i==0:state['base']=imgs
        # Verify the public inference script's eight-call sequence instead of
        # silently assuming a caller always performs the same augmentations.
        assert torch.equal(imgs,transform(state['base'],i)),'Unexpected D4 encode call order'
        if not state['pending']:
            batch=torch.cat([transform(state['base'],j) for j in range(i,min(8,i+batch_size))],0)
            features,logits=original(batch)
            state['pending']=[(features[j:j+1],[v[j:j+1] for v in logits]) for j in range(len(batch))]
        result=state['pending'].pop(0);state['index']=(i+1)%8
        if i==7:state['base']=None;assert not state['pending']
        return result
    model.encode=encode
    return model
