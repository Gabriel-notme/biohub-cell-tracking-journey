from pathlib import Path
import os,json,time,math
os.environ['OMP_NUM_THREADS']='3'
import numpy as np,torch
from torch.nn import functional as F
from centroid_model import CentroidNet
from train_events import atomic_save
R=Path('/workspace/biohub');D=R/'centroid_data';O=R/'Centroid_v1';O.mkdir(exist_ok=True);torch.set_num_threads(3);torch.manual_seed(20261401);np.random.seed(20261401)
def load(names):
    xx=[];yy=[];ff=[]
    for n in names:
        with np.load(D/(n+'.npz')) as d:yy.append(d['offset']);ff.append(d['fraction'])
        xx.append(np.load(D/(n+'_patches.npy')))
    return torch.from_numpy(np.concatenate(xx)).cuda(),torch.from_numpy(np.concatenate(yy)).cuda(),torch.from_numpy(np.concatenate(ff)).cuda()
@torch.inference_mode()
def validate(model,data):
    model.eval();x,y,f=data;errors=[];old=[];expected=[]
    for start in range(0,len(x),256):
        with torch.autocast('cuda',dtype=torch.float16):pred,unc=model(x[start:start+256].float(),f[start:start+256])
        errors.append(torch.linalg.vector_norm(pred.float()-y[start:start+256],dim=1).cpu().numpy());old.append(torch.linalg.vector_norm(y[start:start+256],dim=1).cpu().numpy());expected.append(unc.float().cpu().numpy())
    e=np.concatenate(errors);b=np.concatenate(old);report={'mean_um':float(e.mean()),'median_um':float(np.median(e)),'baseline_mean_um':float(b.mean()),'baseline_median_um':float(np.median(b)),'within_2um':float((e<=2).mean()),'within_3um':float((e<=3).mean()),'n':len(e)};model.train();return report
if __name__=='__main__':
    while not (D/'ready.json').exists():time.sleep(15)
    assert not (O/'best.pt').exists();split=json.loads((D/'split.json').read_text());train=load(split['train']);cal=load(split['calibration']);ck=torch.load(R/'Predicted7_v1_frozen.pt',map_location='cpu',weights_only=False);model=CentroidNet(ck['config']).cuda();model.base.load_state_dict(ck['model']);optimizer=torch.optim.AdamW([{'params':model.base.parameters(),'lr':8e-5},{'params':model.head.parameters(),'lr':3e-4}],weight_decay=.01);config={'architecture':'centroid_offset_regressor','base_config':ck['config']};best=1e30;history=[];start=time.time();x,y,f=train;spacing=x.new_tensor([1.625,.8125,.8125],dtype=torch.float32)
    print('CENTROID_BANK',len(x),len(cal[0]),flush=True)
    for step in range(1,12001):
        ix=torch.randint(len(x),(256,),device='cuda');xx=x[ix].float();yy=y[ix].clone();ff=f[ix].clone()
        for axis in range(3):
            if np.random.rand()<.5:xx=xx.flip(axis+2);yy[:,axis]*=-1;ff[:,axis]=-ff[:,axis]-spacing[axis]
        if np.random.rand()<.5:xx=xx.transpose(-1,-2);yy=yy[:,[0,2,1]];ff=ff[:,[0,2,1]]
        xx=xx*(.75+.5*torch.rand((len(xx),1,1,1,1),device='cuda'))
        lrscale=(.1+.9*.5*(1+math.cos(math.pi*step/12000)))*min(1.,step/200)
        optimizer.param_groups[0]['lr']=8e-5*lrscale;optimizer.param_groups[1]['lr']=3e-4*lrscale;optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            pred,unc=model(xx,ff);distance=torch.linalg.vector_norm(pred.float()-yy,dim=1);loss=F.smooth_l1_loss(pred.float(),yy,beta=1.)+.05*F.smooth_l1_loss(unc.float(),distance.detach())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
        if step%200==0:print('CENTROID_STEP',step,float(loss),round(time.time()-start),flush=True)
        if step%1500==0:
            report=validate(model,cal);report['step']=step;history.append(report);payload={'model':model.state_dict(),'config':config,'calibration':report,'training_movies':split['train'],'step':step}
            if report['mean_um']<best:best=report['mean_um'];atomic_save(payload,O/'best.pt')
            (O/'history.json').write_text(json.dumps(history,indent=2));print('CENTROID_VALIDATION',json.dumps(report),flush=True)
    (O/'done.json').write_text(json.dumps({'best_mean_um':best,'seconds':time.time()-start,'history':history},indent=2));print('CENTROID_TRAIN_DONE',best,flush=True)
