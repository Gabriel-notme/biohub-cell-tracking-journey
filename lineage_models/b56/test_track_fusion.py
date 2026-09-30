from pathlib import Path
import numpy as np,torch,json
from track_video import TrackVideoNet,STEPS,PATCH
from track_fusion import encode_bank,fusion_features

torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(892)
rng=np.random.default_rng(892);reports={}
for channels in [1,3]:
    model=TrackVideoNet(channels=channels).cuda().eval();n=25
    crops=rng.random((n,*((channels,) if channels>1 else ()),*PATCH)).astype(np.float16)
    seq=rng.integers(0,n,(2,n,STEPS));motion=rng.normal(0,.1,(2,n,STEPS,4)).astype(np.float32);motion[:,:,:,3]=1
    bank=encode_bank(model,crops,seq,motion);rr=np.array([[2,5,10],[6,3,21]])
    x=np.stack([crops[np.stack([seq[int(j>0),r[j]] for j in range(3)])] for r in rr]);m=np.stack([np.stack([motion[int(j>0),r[j]] for j in range(3)]) for r in rr]);g=rng.normal(0,1,(2,28)).astype(np.float32)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p,_=model(torch.as_tensor(x,device='cuda'),torch.as_tensor(m,device='cuda'),torch.as_tensor(g,device='cuda'),'fork')
    features=fusion_features(model,bank,rr,g,np.zeros((2,530),np.float32),'fork')
    error=float(np.max(np.abs(p.float().cpu().numpy()-features[:,530])));assert error<.002,error
    swapped=fusion_features(model,bank,rr[:,[0,2,1]],g,np.zeros((2,530),np.float32),'fork');assert np.array_equal(features,swapped)
    assert features.shape==(2,790)
    reports[str(channels)]={'train_vs_banked_logit_max_error':error,'daughter_order_invariant':True,'features':features.shape[1]}
Path('/workspace/biohub/track_fusion_tests.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports))
