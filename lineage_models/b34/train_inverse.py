from pathlib import Path
import os,json,time,random,shutil
os.environ['JOINT_DENSE_MODE']='softmax';os.environ.setdefault('OMP_NUM_THREADS','3')
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from train_joint import augment,dense_loss
from joint_model import JointLineageNet,SHAPE,SPACING
from train_events import atomic_save
R=Path('/workspace/biohub');D=R/'inverse_data';O=R/'Inverse_dense';O.mkdir(exist_ok=True)
class InverseData(Dataset):
    def __init__(self,names):
        self.names=names;self.bank={};rows=[];ys=[];ws=[]
        for m,n in enumerate(names):
            with np.load(D/(n+'.npz')) as f:rows.extend((m,i) for i in range(len(f['target'])));ys.extend(f['target']);ws.extend(f['weight'])
        self.rows=rows;self.y=np.asarray(ys,np.float32);self.weight=np.asarray(ws,np.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        m,j=self.rows[i]
        if m not in self.bank:self.bank[m]=np.load(D/(self.names[m]+'_patches.npy'),mmap_mode='r')
        return np.array(self.bank[m][j]),self.y[i].copy()
@torch.inference_mode()
def validate(model,loader):
    model.eval();dist=[]
    for x,y in loader:
        with torch.autocast('cuda',dtype=torch.float16):heat=model.encode(x.cuda().float())[-2]
        ix=heat[:,0].float().flatten(1).argmax(1).cpu().numpy();coord=(np.stack(np.unravel_index(ix,SHAPE),-1)-np.asarray(SHAPE)//2)*SPACING;dist.extend(np.linalg.norm(coord-y.numpy(),axis=1))
    d=np.asarray(dist);model.train();return {'median_um':float(np.median(d)),'recall_3um':float((d<=3).mean()),'recall_5um':float((d<=5).mean()),'n':len(d)}
if __name__=='__main__':
    while not (D/'ready.json').exists() or not (R/'dense_training_complete.json').exists():time.sleep(15)
    assert not (O/'history.json').exists();torch.set_num_threads(3);torch.manual_seed(20261048);np.random.seed(20261048);random.seed(20261048)
    split=json.loads((D/'split.json').read_text());train=InverseData(split['train']);val=InverseData(split['calibration']);loader=DataLoader(train,batch_size=32,sampler=WeightedRandomSampler(torch.from_numpy(train.weight),max(len(train),32000),replacement=True),num_workers=3,persistent_workers=True,pin_memory=True,drop_last=True);vl=DataLoader(val,batch_size=64,num_workers=2,pin_memory=True)
    ck=torch.load(R/'B3_dense_motion_frozen.pt',map_location='cpu',weights_only=False);model=JointLineageNet(**ck['config']).cuda();model.load_state_dict(ck['model']);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.03);it=iter(loader);start=time.time();best=-1.;history=[]
    for step in range(1,24001):
        try:x,y=next(it)
        except StopIteration:it=iter(loader);x,y=next(it)
        x=x.cuda().float();y=y.cuda();r=torch.stack([torch.zeros_like(y),y],1);children=torch.stack([y,torch.full_like(y,float('nan'))],1);g=x.new_zeros((len(x),16));x,r,g,children=augment(x,r,g,children,'edge',spatial_jitter=2.5)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):heat=model.encode(x,r[:,0])[-2];loss=dense_loss(heat,children)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        for pg in opt.param_groups:pg['lr']=1e-4*(.1+.9*(1+np.cos(np.pi*step/24000))/2)
        if step%100==0:print('INVERSE_STEP',step,'loss',round(loss.item(),5),'seconds',round(time.time()-start),flush=True)
        if step%3000==0 or step==1:
            stats=validate(model,vl);objective=stats['recall_5um']+.5*stats['recall_3um']-.02*stats['median_um'];state={'config':ck['config'],'model':model.state_dict(),'step':step,'validation':stats,'task':'temporal_reverse_dense_parent_correspondence','training_movies':split['train']};atomic_save(state,O/'last.pt')
            if objective>best:best=objective;atomic_save(state,O/'best.pt')
            history.append({'step':step,'validation':stats,'objective':objective});(O/'history.json').write_text(json.dumps(history,indent=2));print('INVERSE_VALIDATION',step,json.dumps(stats),flush=True)
    shutil.copy2(O/'best.pt',R/'Inverse_dense_frozen.pt');(O/'done.json').write_text(json.dumps({'step':step,'seconds':time.time()-start,'best':best}));print('INVERSE_COMPLETE',flush=True)
