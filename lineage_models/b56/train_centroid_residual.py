from pathlib import Path
import os,json,time,math,argparse,hashlib
os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np,torch
from torch.nn import functional as F
from centroid_model import CentroidNet
R=Path('/workspace/biohub');D=R/'centroid_residual_data'
def load(names):
    xx=[];yy=[];ff=[]
    for n in names:
        with np.load(D/(n+'.npz')) as d:yy.append(d['offset']);ff.append(d['fraction'])
        xx.append(np.load(D/(n+'_patches.npy')))
    return torch.from_numpy(np.concatenate(xx)).cuda(),torch.from_numpy(np.concatenate(yy)).cuda(),torch.from_numpy(np.concatenate(ff)).cuda()
@torch.inference_mode()
def validate(model,data):
    model.eval();x,y,f=data;errors=[]
    for start in range(0,len(x),192):
        with torch.autocast('cuda',dtype=torch.float16):pred,_=model(x[start:start+192].float(),f[start:start+192])
        errors.append(torch.linalg.vector_norm(pred.float()-y[start:start+192],dim=1).cpu().numpy())
    e=np.concatenate(errors);return {'mean_um':float(e.mean()),'median_um':float(np.median(e)),'within_3um':float((e<=3).mean()),'n':len(e),'baseline_mean_um':float(torch.linalg.vector_norm(y,dim=1).mean())}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();O=R/('TrackCentroid_'+str(a.seed));O.mkdir(exist_ok=True)
    assert not (O/'best.pt').exists();torch.set_num_threads(3);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.backends.cudnn.benchmark=True
    split=json.loads((D/'split.json').read_text());train=load(split['train']);cal=load(split['calibration']);ck=torch.load(R/'centroid_v1_frozen.pt',map_location='cpu',weights_only=False)
    model=CentroidNet(ck['config']['base_config']).cuda();model.load_state_dict(ck['model'])
    with torch.no_grad():model.head[-1].weight.zero_();model.head[-1].bias.zero_();model.head[-1].bias[-1]=1.8
    optimizer=torch.optim.AdamW([{'params':model.base.parameters(),'lr':2e-5},{'params':model.head.parameters(),'lr':2e-4}],weight_decay=.02)
    protocol={'train':split['train'],'calibration':split['calibration'],'new_audit_excluded':split['new_audit'],'initialization':'Inherited B4 seven-frame centroid encoder; displacement head reset for residual prediction on actual B4 coordinates. Inherited encoder training overlaps held-out movies.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (O/'training_protocol.json').write_text(json.dumps(protocol,indent=2));best=1e30;history=[];start=time.time();x,y,f=train;spacing=x.new_tensor([1.625,.8125,.8125],dtype=torch.float32)
    for step in range(1,6001):
        if time.time()-start>2400:break
        ix=torch.randint(len(x),(192,),device='cuda');xx=x[ix].float();yy=y[ix].clone();ff=f[ix].clone()
        for axis in range(3):
            if np.random.rand()<.5:xx=xx.flip(axis+2);yy[:,axis]*=-1;ff[:,axis]=-ff[:,axis]-spacing[axis]
        if np.random.rand()<.5:xx=xx.transpose(-1,-2);yy=yy[:,[0,2,1]];ff=ff[:,[0,2,1]]
        xx=xx*(.6+.8*torch.rand((len(xx),1,1,1,1),device='cuda'))
        lrscale=(.1+.9*.5*(1+math.cos(math.pi*min(1,max(step/6000,(time.time()-start)/2400)))))*min(1.,step/100)
        optimizer.param_groups[0]['lr']=2e-5*lrscale;optimizer.param_groups[1]['lr']=2e-4*lrscale;optimizer.zero_grad(set_to_none=True);model.train()
        with torch.autocast('cuda',dtype=torch.float16):
            pred,unc=model(xx,ff);distance=torch.linalg.vector_norm(pred.float()-yy,dim=1);loss=F.smooth_l1_loss(pred.float(),yy,beta=1.)+.05*F.smooth_l1_loss(unc.float(),distance.detach())+.01*pred.float().square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
        if step%200==0:print('CENTROID_RESIDUAL_STEP',step,float(loss.detach()),round(time.time()-start),flush=True)
        if step%600==0:
            report=validate(model,cal);report['step']=step;history.append(report)
            if report['mean_um']<best:
                best=report['mean_um'];torch.save({'model':model.state_dict(),'config':ck['config'],'protocol':protocol,'calibration':report,'step':step},O/'best.pt')
            (O/'history.json').write_text(json.dumps(history,indent=2));print('CENTROID_RESIDUAL_VALIDATION',json.dumps(report),flush=True)
    assert (O/'best.pt').exists();(O/'ready.json').write_text(json.dumps({'seconds':time.time()-start,'best_mean_um':best}));print('CENTROID_RESIDUAL_DONE',flush=True)
