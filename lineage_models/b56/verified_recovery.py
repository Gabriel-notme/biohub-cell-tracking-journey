"""Independent temporal appearance confirmation of image-supported recovered nodes."""
from pathlib import Path
import os,json,hashlib
import numpy as np,torch
from recover_joint import RecoveryRefiner,RECOVERY_DEFAULT,bridge_future
from transfer_model import load_transfer
from cell_event import Movie,SCALE,chain,edge_geometry,fork_geometry
from refine_events import structure

class VerifiedRecoveryRefiner(RecoveryRefiner):
    def __init__(self,paths,root,config=None,cache_dir=None):
        super().__init__(paths[:1],root,config,cache_dir);self.verifier,_=load_transfer(paths[1]);self.paths=[Path(p) for p in paths]
        torch.backends.mha.set_fastpath_enabled(False)
        self.recovery_signature=self.signature
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    @torch.inference_mode()
    def proposals(self,name,nodes,edges):
        signature=self.signature;self.signature=self.recovery_signature
        try:proposals=super().proposals(name,nodes,edges)
        finally:self.signature=signature
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        candidate_config={k:self.config[k] for k in RECOVERY_DEFAULT}
        candidate_sha=hashlib.sha256(json.dumps(candidate_config,sort_keys=True).encode()).hexdigest()[:12]
        path=self.cache/('verified_'+name+'_'+self.signature+'_'+graphsha+'_'+candidate_sha+'.json')
        if path.exists():result=json.loads(path.read_text())
        else:
            out,prev,_,pos=structure(nodes,edges)
            # These exact same gates are applied by RecoveryRefiner.refine;
            # verifying a proposal that cannot pass them has no effect on output.
            chosen=[r for r in proposals if r['heat']>=self.config['heat_threshold'] and r['probability']>=self.config['fork_threshold' if r['kind']=='fork' else 'edge_threshold']]
            requests=[];lookup={};rows=[];geometry=[]
            def request(t,coord):
                key=(int(t),*np.rint(np.asarray(coord)*32).astype(int).tolist())
                if key not in lookup:lookup[key]=len(requests);requests.append((t,coord))
                return lookup[key]
            for r in chosen:
                s,d=r['s'],r['d'];t=int(nodes[s]['t']);q=np.asarray(r['position']);h=chain(s,prev,pos);ff=bridge_future(q,d,out,pos,int(self.config['bridge_gap']))
                ids=[request(t,pos[s]/SCALE)]
                if r['kind']=='fork':
                    a=r['old'][0];ids.append(request(t+1,pos[a]/SCALE));g=fork_geometry(h,chain(a,out,pos),ff)
                else:g=edge_geometry(h,ff)
                ids.append(request(t+1,q/SCALE));rows.append(ids);geometry.append(g)
            bank=torch.empty((len(requests),3,64),device='cuda',dtype=torch.float32);movie=Movie(self.root/(name+'.zarr'),context=7);order=sorted(range(len(requests)),key=lambda i:requests[i][0])
            for start in range(0,len(order),192):
                indices=order[start:start+192];req=[requests[i] for i in indices];x=torch.from_numpy(movie.patches([t for t,c in req],[c for t,c in req])).cuda().float()
                with torch.autocast('cuda',dtype=torch.float16):z=self.verifier.base.encode(torch.cat([x[:,0:3],x[:,2:5],x[:,4:7]],0))
                bank[indices]=z.reshape(3,len(req),64).transpose(0,1).float()
            if len(requests):
                for kind in ['edge','fork']:
                    indices=[i for i,r in enumerate(chosen) if r['kind']==kind]
                    for start in range(0,len(indices),512):
                        ii=indices[start:start+512];ix=torch.tensor([rows[i] for i in ii],device='cuda');g=torch.from_numpy(np.stack([geometry[i] for i in ii])).cuda()
                        with torch.autocast('cuda',dtype=torch.float16):p=self.verifier(bank[ix],g,kind)
                        for i,v in zip(ii,p.float().sigmoid().cpu().tolist()):chosen[i]['verification_probability']=float(v)
            result=chosen;tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(result));os.replace(tmp,path)
        threshold=self.config.get('verification_threshold',.5)
        return [r for r in result if r.get('verification_probability',0)>=threshold]
