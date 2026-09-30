from pathlib import Path
import os,json,time,argparse,hashlib
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('POLARS_MAX_THREADS','2')
import numpy as np,torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from track_set import TrackSetNet,mask_incumbent,event_probabilities
from grouped_validation import grouped_mask
R=Path('/workspace/biohub');STRUCTURED=True

def load_data(variant,split,group):
    xx=[];yy=[];masks=[];groups=[];importance=[];canonical=sorted(set(split.get('augmentation_source_movies',{}).get(n,n) for n in split['train']));cap=12
    for name in split[group]:
        actual=split.get('augmentation_source_movies',{}).get(name,name)
        if variant=='Set3':
            with np.load(R/'TrackSet3_features'/(name+'.npz')) as d:x=d['edge'];y=d['edge_y'];source=d['source']
        else:
            with np.load(R/'Track3_fusion/features'/(name+'.npz')) as a,np.load(R/'Track_fusion/features'/(name+'.npz')) as b:x=np.c_[a['edge'],b['edge'][:,338:]];y=a['edge_y'];assert np.array_equal(y,b['edge_y'])
            with np.load(R/'track_b4w_3'/(name+'.npz')) as d:source=d['edge'][:,0]
        for parent in np.unique(source):
            ix=np.flatnonzero(source==parent)
            if len(ix)>cap:raise ValueError('Candidate truncation is prohibited')
            if y[ix].sum()>2:continue
            block=np.zeros((cap,x.shape[1]),np.float16);target=np.zeros(cap,np.float32);valid=np.zeros(cap,bool)
            block[:len(ix)]=x[ix];target[:len(ix)]=y[ix];valid[:len(ix)]=True
            old=x[ix,45]>.5;error=bool(np.any(old&(y[ix]==0)) or np.any((~old)&(y[ix]>0)))
            weight=(12. if error else 1.)*(4. if target.sum()==2 else 1.)
            xx.append(block);yy.append(target);masks.append(valid);groups.append(canonical.index(actual) if group=='train' else 0);importance.append(weight)
    return tuple(torch.from_numpy(np.asarray(v)).cuda() for v in [xx,yy,masks,groups,importance]),canonical

def loss_fn(outputs,y,valid,weights=None):
    edge,count,pair=outputs;n=y.sum(1).long();bce=(F.binary_cross_entropy_with_logits(edge.float(),y,reduction='none')*valid).sum(1)/valid.sum(1)
    null=count[:,0:1].float();distribution=torch.cat([edge.float(),null],1).log_softmax(1);target=y/n.clamp_min(1)[:,None]
    ranking=-(distribution[:,:-1]*target).sum(1);ranking=torch.where(n>0,ranking,-distribution[:,-1])
    countloss=F.cross_entropy(count.float(),n,reduction='none');pairloss=torch.zeros_like(bce);positive=n==2
    if positive.any():
        yy=y[positive].bool();truth=(yy[:,:,None]&yy[:,None,:]&torch.ones((y.shape[1],y.shape[1]),device=y.device,dtype=torch.bool).triu(1)).flatten(1).float().argmax(1)
        pairloss[positive]=F.cross_entropy(pair[positive].float().flatten(1),truth,reduction='none')
    if STRUCTURED:
        conditional=-(edge.float().log_softmax(1)*y).sum(1)
        loss=countloss+torch.where(n==1,conditional,torch.where(n==2,pairloss,torch.zeros_like(bce)))+.2*bce
    else:loss=bce+.75*ranking+.5*countloss+.75*pairloss
    return (loss*weights).mean() if weights is not None else loss.mean()

@torch.inference_mode()
def evaluate(model,data,indices):
    x,y,valid,_,_=data;model.eval();losses=[];labels=[];pred=[];count_correct=0;classloss={k:[] for k in range(3)}
    for start in range(0,len(indices),512):
        ix=indices[start:start+512]
        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(mask_incumbent(x[ix].float()),valid[ix])
        losses.append(float(loss_fn(outputs,y[ix],valid[ix]))*len(ix));p,_=event_probabilities(*outputs,valid[ix],structured=STRUCTURED);labels.extend(y[ix][valid[ix]].cpu().numpy());pred.extend(p[valid[ix]].cpu().numpy());count_correct+=int((outputs[1].argmax(1)==y[ix].sum(1)).sum())
        for k in range(3):
            chosen=y[ix].sum(1)==k
            if chosen.any():classloss[k].append((float(loss_fn(tuple(q[chosen] for q in outputs),y[ix][chosen],valid[ix][chosen])),int(chosen.sum())))
    byclass={k:sum(v*n for v,n in entries)/sum(n for v,n in entries) for k,entries in classloss.items() if entries}
    return {'loss':sum(losses)/len(indices),'balanced_loss':sum(byclass[k]*({0:.25,1:.5,2:.25}[k]) for k in byclass)/sum({0:.25,1:.5,2:.25}[k] for k in byclass),'class_loss':byclass,'edge_ap':float(average_precision_score(labels,pred)),'count_accuracy':count_correct/len(indices)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['Set3','SetDual'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=6000);a=p.parse_args();torch.set_num_threads(3);torch.manual_seed(a.seed);torch.backends.mha.set_fastpath_enabled(False)
    O=R/('TrackStructured'+a.variant.removeprefix('Set')+'_'+str(a.seed));O.mkdir(exist_ok=True);assert not (O/'ready.json').exists();split=json.loads((R/('TrackSet3_features' if a.variant=='Set3' else 'track_data')/'split.json').read_text())
    data,names=load_data(a.variant,split,'train');x,y,valid,group,importance=data
    held,fold=grouped_mask(names,group.cpu().numpy(),(y.sum(1)==2).cpu().numpy().astype(int),a.seed)
    assert fold['usable'],'A grouped internal validation with divisions is required'
    hold=torch.as_tensor(held,device='cuda');trainix=torch.where(~hold)[0];validix=torch.where(hold)[0];config={'input_dim':x.shape[-1],'width':96,'depth':2,'architecture':'candidate_set_attention','structured':True}
    encoders=['TrackRobust_137/best.pt'] if a.variant=='Set3' else ['Track3_b4_137/best.pt','Track_b4_137/best.pt']
    protocol={'train':split['train'],'calibration':split['calibration'],'new_audit_excluded':split['new_audit'],'augmentation_source_movies':split.get('augmentation_source_movies',{}),'internal_fold':fold,'training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model':'Structured complete-event likelihood and posterior marginals; candidate-set self-attention; competing continuations, child-count classification, symmetric daughter-pair likelihood; group-held-out loss selects steps then full-training refit. All inherited graph probability and membership inputs are masked.'}
    (O/'training_protocol.json').write_text(json.dumps(protocol,indent=2));history=[];started=time.time();best=float('inf');beststep=a.steps
    for phase in ['internal','refit']:
        torch.manual_seed(a.seed);model=TrackSetNet(**config).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.04);indices=trainix if phase=='internal' else torch.arange(len(x),device='cuda');steps=a.steps if phase=='internal' else beststep;sampling=importance[indices].float()
        for step in range(1,steps+1):
            ix=indices[torch.multinomial(sampling,128,replacement=True)];xx=mask_incumbent(x[ix].float());xx+=torch.randn_like(xx)*.025;model.train();optimizer.zero_grad(set_to_none=True);optimizer.param_groups[0]['lr']=4e-4*(.1+.9*(1+np.cos(np.pi*step/steps))/2)*min(1,step/100)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(model(xx,valid[ix]),y[ix],valid[ix])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),4.);optimizer.step()
            if step%200==0:print('SET_TRAIN',phase,step,round(float(loss.detach()),5),round(time.time()-started),flush=True)
            if phase=='internal' and step%1000==0:
                metrics=evaluate(model,data,validix);history.append({'phase':phase,'step':step,**metrics});(O/'history.json').write_text(json.dumps(history,indent=2));print('SET_INTERNAL_VALIDATION',json.dumps(history[-1]),flush=True)
                if metrics['balanced_loss']<best:best=metrics['balanced_loss'];beststep=step
        if phase=='internal':protocol['refit_steps']=beststep
    checkpoint={'config':config,'model':model.state_dict(),'protocol':protocol,'encoder_sha256_list':[hashlib.sha256((R/n).read_bytes()).hexdigest() for n in encoders],'encoders':encoders,'seed':a.seed}
    torch.save(checkpoint,O/'best.pt');(O/'training_protocol.json').write_text(json.dumps(protocol,indent=2));(O/'ready.json').write_text(json.dumps({'seconds':time.time()-started,'refit_steps':beststep}));print('SET_MODEL_DONE',flush=True)
