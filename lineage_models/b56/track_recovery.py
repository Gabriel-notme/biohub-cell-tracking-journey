"""Verify image-supported missing-cell proposals with newly trained long tracklets."""
from pathlib import Path
import hashlib,json,os
import numpy as np,torch
from scipy.special import expit
from recover_joint import RecoveryRefiner,RECOVERY_DEFAULT
from track_video import load_track,TrackMovie,walk,STEPS
from refine_events import structure
from cell_event import SCALE,edge_geometry,fork_geometry

class TrackRecoveryRefiner(RecoveryRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        super().__init__(paths[:1],root,config,cache_dir)
        self.proposal_signature=self.signature;self.paths=[Path(p) for p in paths]
        self.track_models=[load_track(p)[0] for p in paths[1:]]
        assert self.track_models and len({m.channels for m in self.track_models})==1
        assert all(m.edge_dim==16 and m.fork_dim==28 for m in self.track_models)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def proposals(self,name,nodes,edges):
        sig=self.signature;self.signature=self.proposal_signature
        try:raw=super().proposals(name,nodes,edges)
        finally:self.signature=sig
        cfg=self.config;key=hashlib.sha256(json.dumps([nodes,edges,cfg],sort_keys=True).encode()).hexdigest()[:16]
        path=self.cache/f'track_recovery_{name}_{sig}_{key}.json'
        if path.exists():return json.loads(path.read_text())
        chosen=[dict(r) for r in raw if r['heat']>=cfg['heat_threshold'] and r['probability']>=cfg.get('proposal_min_probability',.25) and (cfg['allow_division'] if r['kind']=='fork' else cfg['allow_gap'])]
        if not chosen:path.write_text('[]');return []
        out,prev,frames,pos=structure(nodes,edges);requests=[];lookup={};events=[];event_owner=[]
        def original_chain(n,links):return [(int(nodes[q]['t']),pos[q]) for q in walk(n,links)]
        def request(t,p):
            identity=(int(t),*np.rint(np.asarray(p)*64).astype(int).tolist())
            if identity not in lookup:lookup[identity]=len(requests);requests.append((int(t),np.asarray(p)))
            return lookup[identity]
        def track(items):
            items=items[:STEPS];indices=[request(t,p) for t,p in items];points=np.stack([p for t,p in items]);n=len(items)
            indices=np.asarray(indices+[indices[-1]]*(STEPS-n),np.int64)
            m=np.zeros((STEPS,4),np.float32);m[:n,:3]=(points-points[:1])/10;m[n:,:3]=m[n-1,:3];m[:n,3]=1
            return indices,m
        for ri,r in enumerate(chosen):
            s,d=r['s'],r['d'];gap=int(cfg['bridge_gap']);t=int(nodes[s]['t']);q=np.asarray(r['position']);dest=pos[d]
            middle=[(t+1+j,q+(dest-q)*j/gap) for j in range(gap)]
            history=original_chain(s,prev);future=original_chain(d,out);forward=middle+future
            trajectories=[history,forward];kind=r['kind']
            if kind=='fork':trajectories=[history,original_chain(r['old'][0],out),forward]
            points=[np.stack([p for t,p in tr[:4]]) for tr in trajectories]
            g=edge_geometry(*points) if kind=='edge' else fork_geometry(*points)
            events.append((kind,[track(tr) for tr in trajectories],g));event_owner.append((ri,'event'))
            for j in range(gap):
                hh=list(reversed(middle[:j+1]))+history;ff=middle[j+1:]+future
                gg=edge_geometry(np.stack([p for t,p in hh[:4]]),np.stack([p for t,p in ff[:4]]))
                events.append(('edge',[track(hh),track(ff)],gg));event_owner.append((ri,'arrival'))
        movie=TrackMovie(self.root/(name+'.zarr'),self.track_models[0].channels);features=[np.empty((len(requests),m.width),np.float32) for m in self.track_models]
        order=sorted(range(len(requests)),key=lambda i:requests[i][0])
        for start in range(0,len(order),192):
            ii=order[start:start+192];x=np.stack([movie.crops(t,[p/SCALE])[0] for t,p in [requests[i] for i in ii]])
            x=torch.as_tensor(x,device='cuda').float()
            if self.track_models[0].channels==1:x=x[:,None]
            for mi,model in enumerate(self.track_models):
                with torch.autocast('cuda',dtype=torch.float16):z=model.encode_frame(x)
                features[mi][ii]=z.float().cpu().numpy()
        logits=np.empty((len(self.track_models),len(events)),np.float32)
        for kind in ['edge','fork']:
            indices=[i for i,e in enumerate(events) if e[0]==kind]
            for start in range(0,len(indices),256):
                ii=indices[start:start+256];chosen_events=[events[i] for i in ii];ix=np.stack([[tr[0] for tr in e[1]] for e in chosen_events]);mm=np.stack([[tr[1] for tr in e[1]] for e in chosen_events]);gg=torch.as_tensor(np.stack([e[2] for e in chosen_events]),device='cuda')
                b,k,t=ix.shape;m=torch.as_tensor(mm,device='cuda').reshape(b*k,t,4)
                for mi,(model,bank) in enumerate(zip(self.track_models,features)):
                    z=torch.as_tensor(bank[ix],device='cuda').reshape(b*k,t,-1)
                    with torch.autocast('cuda',dtype=torch.float16):
                        z=model.encode_track(z,m).reshape(b,k,-1);p=model.classify(z,gg,kind)
                    logits[mi,ii]=p.float().cpu().numpy()
        probabilities=expit(logits.mean(0));result=[]
        for ri,r in enumerate(chosen):
            main=[float(p) for p,o in zip(probabilities,event_owner) if o==(ri,'event')][0]
            bridge=[float(p) for p,o in zip(probabilities,event_owner) if o==(ri,'arrival')]
            if min(bridge)<cfg.get('arrival_threshold',.95):continue
            r.update(legacy_probability=r['probability'],probability=main,arrival_probability=min(bridge));result.append(r)
        tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(result));os.replace(tmp,path)
        return result
