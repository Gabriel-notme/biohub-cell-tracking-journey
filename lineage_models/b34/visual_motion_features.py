import numpy as np,torch

@torch.inference_mode()
def visual_edge_features(motion,rows,models,banks,lookup):
    indices=np.asarray([[lookup[int(n)] for n in row] for row in rows]);parts=[motion]
    for model,bank in zip(models,banks):
        chunks=[]
        for start in range(0,len(rows),2048):
            ix=indices[start:start+2048];z=torch.from_numpy(bank[ix]).cuda().float();g=torch.from_numpy(motion[start:start+len(ix),:16]).cuda()
            p,c=z.unbind(1);logits=model.edge_logits(p,c,g);phase=model.phase(z).squeeze(-1)
            feature=torch.cat([logits[:,None],phase,p,c,(p-c).abs(),p*c],1)
            chunks.append(feature.float().cpu().numpy())
        parts.append(np.concatenate(chunks))
    return np.concatenate(parts,axis=1).astype(np.float32)

@torch.inference_mode()
def visual_fork_features(motion,rows,models,banks,lookup):
    indices=np.asarray([[lookup[int(n)] for n in row] for row in rows]);parts=[motion];geometry=motion[:,:28]
    for model,bank in zip(models,banks):
        chunks=[]
        for start in range(0,len(rows),2048):
            ix=indices[start:start+2048];z=torch.from_numpy(bank[ix]).cuda().float();g=torch.from_numpy(geometry[start:start+len(ix)]).cuda()
            p,a,b=z.unbind(1);logits=model.fork_logits(p,a,b,g);phase=model.phase(z).squeeze(-1)
            feature=torch.cat([logits[:,None],phase[:,0:1],phase[:,1:].amin(1,keepdim=True),phase[:,1:].amax(1,keepdim=True),p,(a+b)*.5,(a-b).abs()],1)
            chunks.append(feature.float().cpu().numpy())
        parts.append(np.concatenate(chunks))
    return np.concatenate(parts,axis=1).astype(np.float32)
