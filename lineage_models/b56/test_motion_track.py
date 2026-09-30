from pathlib import Path
import json,torch,numpy as np
from track_video import TrackVideoNet,load_track
from train_track_video import Events
R=Path('/workspace/biohub');D=R/'track_motion3';names=json.loads((D/'split.json').read_text())['train'];name=next(n for n in names if (D/(n+'.npz')).exists());torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
model=TrackVideoNet(channels=3,edge_dim=338,fork_dim=530).cuda();results={}
for kind in ['edge','fork']:
    ds=Events(D,[name],kind);x,m,g,y,p=ds[0];x=torch.as_tensor(x[None],device='cuda');m=torch.as_tensor(m[None],device='cuda');g=torch.as_tensor(g[None],device='cuda')
    with torch.autocast('cuda',dtype=torch.float16):a,b=model(x,m,g,kind);loss=a.float().square().mean()+b.float().square().mean()
    assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None);model.zero_grad(set_to_none=True);results[kind]={'geometry_columns':g.shape[1],'finite_forward_backward':True}
old,_=load_track(R/'Track3_full_137/best.pt');z=torch.randn(3,2,64,device='cuda');g=torch.randn(3,16,device='cuda')
with torch.inference_mode():
    p,a=z[:,0],z[:,1];expected=old.edge(torch.cat([p,a,(p-a).abs(),p*a,g],1)).squeeze(1);got=old.classify(z,g,'edge');assert torch.equal(expected,got)
results['legacy_classifier_identical']=True;(R/'track_hybrid_tests.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
