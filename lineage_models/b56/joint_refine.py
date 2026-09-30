"""Sparse joint event assignment; inference uses images, predictions, and trained weights only."""
from pathlib import Path
import os,json,time,hashlib
from itertools import combinations
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from scipy.special import expit,logit
from joint_model import JointMovie,load_joint
from refine_events import structure
from cell_event import SCALE,chain,edge_geometry,fork_geometry

JOINT_DEFAULT={'edge_weight':1.,'fork_weight':1.,'edge_change_penalty':.5,'fork_change_penalty':1.,'fork_threshold':.95,'fork_veto':.10,'max_distance':14.,'max_candidates':4,'fork_max_distance':13.,'fork_min_branch':2,'motion_weight':.10,'temperature':1.,'new_fork_min_edge':.50,'mode':'joint','solver_seconds':.3,'preserve_inherited_divisions':True,'endpoint_recovery':False,'endpoint_probability':.99,'edge_logit':False}

class JointRefiner:
    def __init__(self,paths,root,config=None,cache_dir=None):
        self.paths=[Path(p) for p in paths];self.models=[load_joint(p)[0] for p in self.paths];self.root=Path(root)
        self.config={**JOINT_DEFAULT,**(config or {})};self.cache=Path(cache_dir or '/kaggle/working/joint_cache');self.cache.mkdir(parents=True,exist_ok=True)
        self.signature=hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in self.paths)).hexdigest()[:16]
    def candidates(self,nodes,edges):
        out,prev,frames,pos=structure(nodes,edges);edge=[];fork=[];eg=[];fg=[]
        for t,sources in sorted(frames.items()):
            targets=frames.get(t+1,[])
            if not targets:continue
            tree=cKDTree([pos[d] for d in targets])
            for s in sources:
                old=out.get(s,[]);h=chain(s,prev,pos);velocity=(h[0]-h[-1])/max(1,len(h)-1)
                near=[targets[i] for i in tree.query_ball_point(pos[s],self.config['max_distance'])]
                near.sort(key=lambda d:np.linalg.norm(pos[d]-pos[s]-.5*velocity))
                cand=list(dict.fromkeys(old+near[:self.config['max_candidates']]))
                if not cand:continue
                for d in cand:edge.append((s,d));eg.append(edge_geometry(h,chain(d,out,pos)))
                eligible=[d for d in cand if np.linalg.norm(pos[d]-pos[s])<=self.config['fork_max_distance'] and len(chain(d,out,pos))>=self.config['fork_min_branch']]
                pairs={tuple(sorted(ds)) for ds in combinations(eligible,2) if np.linalg.norm(pos[ds[0]]-pos[ds[1]])<=20}
                if len(old)==2:pairs.add(tuple(sorted(old)))
                for a,b in sorted(pairs):fork.append((s,a,b));fg.append(fork_geometry(h,chain(a,out,pos),chain(b,out,pos)))
        return edge,fork,np.asarray(eg,np.float32).reshape(-1,16),np.asarray(fg,np.float32).reshape(-1,28)
    @torch.inference_mode()
    def predict(self,name,nodes,edges):
        graphsha=hashlib.sha256(json.dumps([nodes,edges],sort_keys=True).encode()).hexdigest()[:16]
        candidate_cfg={k:self.config[k] for k in ['max_distance','max_candidates','fork_max_distance','fork_min_branch']}
        if self.config.get('record_dense',False):candidate_cfg['record_dense']=True
        cfgsha=hashlib.sha256(json.dumps(candidate_cfg,sort_keys=True).encode()).hexdigest()[:10]
        path=self.cache/(name+'_'+self.signature+'_'+graphsha+'_'+cfgsha+'.npz')
        if path.exists():
            with np.load(path) as d:return {k:d[k] for k in d.files}
        edge,fork,eg,fg=self.candidates(nodes,edges)
        edge=np.asarray(edge,np.int64).reshape(-1,2);fork=np.asarray(fork,np.int64).reshape(-1,3)
        bysource={}
        for kind,rows in [('edge',edge),('fork',fork)]:
            for i,row in enumerate(rows):bysource.setdefault(int(row[0]),{'edge':[],'fork':[]})[kind].append(i)
        sources=sorted(bysource,key=lambda s:(nodes[s]['t'],s));movie=JointMovie(self.root/(name+'.zarr'),max(m.context for m in self.models))
        results={kind:np.empty((len(self.models),len(rows)),np.float32) for kind,rows in [('edge',edge),('fork',fork)]}
        if self.config.get('record_dense',False):results['heat_edge']=np.empty((len(self.models),len(edge)),np.float32)
        for start in range(0,len(sources),32):
            batch=sources[start:start+32];lookup={s:i for i,s in enumerate(batch)}
            x=torch.from_numpy(movie.patches([nodes[s]['t'] for s in batch],[[nodes[s][k] for k in ['z','y','x']] for s in batch])).cuda().float()
            queries={s:list(dict.fromkeys([s]+[int(d) for i in bysource[s]['edge'] for d in edge[i,1:]]+[int(d) for i in bysource[s]['fork'] for d in fork[i,1:]])) for s in batch}
            qindex={s:{n:i for i,n in enumerate(ns)} for s,ns in queries.items()};maxq=max(map(len,queries.values()))
            coordinates=np.zeros((len(batch),maxq,3),np.float32)
            for bi,s in enumerate(batch):
                for qi,n in enumerate(queries[s]):coordinates[bi,qi]=[(nodes[n][k]-nodes[s][k])*float(scale) for k,scale in zip(['z','y','x'],SCALE)]
            qt=torch.from_numpy(coordinates).cuda()
            for mi,model in enumerate(self.models):
                with torch.autocast('cuda',dtype=torch.float16):
                    enc=model.encode(x)
                    pooled=model.pool(enc,qt)
                    if self.config.get('record_dense',False):
                        from joint_model import SHAPE,SPACING
                        shape=qt.new_tensor(SHAPE);spacing=qt.new_tensor(SPACING);grid=((qt/spacing+shape.floor_divide(2))/(shape-1)*2-1).flip(-1).reshape(len(qt),-1,1,1,3)
                        with torch.autocast('cuda',enabled=False):
                            hm=F.max_pool3d(enc[-2].float(),kernel_size=(3,5,5),stride=1,padding=(1,2,2))
                            heat_pooled=F.grid_sample(hm,grid.float(),mode='bilinear',padding_mode='border',align_corners=True)[:,0,:,0,0]
                    for kind,rows,geom in [('edge',edge,eg),('fork',fork,fg)]:
                        indices=[i for s in batch for i in bysource[s][kind]]
                        if not indices:continue
                        ix=np.asarray(indices);chosen=rows[ix];srcidx=torch.tensor([lookup[int(r[0])] for r in chosen],device='cuda')
                        feature_indices=torch.tensor([[qindex[int(r[0])][int(n)] for n in r] for r in chosen],device='cuda')
                        features=pooled[srcidx[:,None],feature_indices]
                        p=model.classify_features(features,torch.from_numpy(geom[ix]).cuda(),kind)
                        results[kind][mi,ix]=p.float().cpu().numpy()
                        if kind=='edge' and self.config.get('record_dense',False):results['heat_edge'][mi,ix]=heat_pooled[srcidx,feature_indices[:,1]].cpu().numpy()
            if start%2048==0:print('JOINT_INFER',name,start,'/',len(sources),flush=True)
        result={'edge':edge,'fork':fork,'edge_logits':results['edge'],'fork_logits':results['fork']}
        if 'heat_edge' in results:result['heat_edge_logits']=results['heat_edge']
        tmp=path.with_name(path.stem+'.'+str(os.getpid())+'.tmp.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path)
        return result
    def refine(self,name,nodes,edges):
        start=time.time();cfg=self.config;pred=self.predict(name,nodes,edges)
        def ensemble(kind):
            value=pred[kind+'_logits'];which=cfg.get(kind+'_expert','mean')
            return value.mean(0) if which=='mean' else value[int(which)]
        ep=expit(ensemble('edge')/cfg['temperature']);fp=expit(ensemble('fork')/cfg['temperature'])
        eq={tuple(map(int,row)):float(p) for row,p in zip(pred['edge'],ep)};fq={tuple(map(int,row)):float(p) for row,p in zip(pred['fork'],fp)}
        member_ep=expit(pred.get('edge_member_logits',pred['edge_logits'])/cfg['temperature'])
        eq_low={tuple(map(int,row)):float(p) for row,p in zip(pred['edge'],member_ep.min(0))}
        eq_high={tuple(map(int,row)):float(p) for row,p in zip(pred['edge'],member_ep.max(0))}
        out,prev,frames,pos=structure(nodes,edges);original={(int(e['source_id']),int(e['target_id'])):dict(e) for e in edges}
        protected=set()
        if cfg.get('protected_division_radius',0)>0:
            for s,ds in out.items():
                if len(ds)!=2:continue
                protected.add(s);cur=s;frontier=[s]
                for _ in range(int(cfg['protected_division_radius'])):
                    if cur in prev:cur=prev[cur];protected.add(cur)
                    frontier=[n for p in frontier for n in out.get(p,[])];protected.update(frontier)
        bysource={}
        for (s,d),p in eq.items():bysource.setdefault(s,[]).append(d)
        fork_source={}
        for (s,a,b),p in fq.items():fork_source.setdefault(s,[]).append((a,b,p))
        reassign=set();release_targets=set()
        if cfg.get('endpoint_reassignment',False):
            for s,ds in bysource.items():
                if out.get(s) or s in protected:continue
                h=chain(s,prev,pos);v=(h[0]-h[-1])/max(1,len(h)-1)
                if len(h)<cfg.get('endpoint_min_history',3):continue
                for d in ds:
                    owner=prev.get(d)
                    if owner is None or owner in protected or len(out.get(owner,[]))!=1:continue
                    if eq_low.get((s,d),0.)<cfg['endpoint_probability'] or eq_high.get((owner,d),1.)>cfg.get('endpoint_old_max',.2):continue
                    if len(chain(d,out,pos))<cfg.get('endpoint_min_future',3):continue
                    if np.linalg.norm(pos[d]-pos[s]-v)>cfg.get('endpoint_motion_error',4.5) or np.linalg.norm(pos[d]-pos[s])>cfg.get('endpoint_max_distance',8.):continue
                    reassign.add((s,d));release_targets.add(d)
        division_reassign=set()
        if cfg.get('division_reassignment',False):
            for s,pairs in fork_source.items():
                old=out.get(s,[])
                if len(old)!=1 or s in protected:continue
                for a,b,p in pairs:
                    if old[0] not in (a,b) or p<cfg['fork_threshold']:continue
                    d=b if a==old[0] else a;owner=prev.get(d)
                    if owner is None or owner in protected or len(out.get(owner,[]))!=1:continue
                    if min(eq_low.get((s,a),0.),eq_low.get((s,b),0.))<cfg['new_fork_min_edge']:continue
                    if eq_high.get((owner,d),1.)>cfg.get('division_old_max',.2):continue
                    division_reassign.add((s,d));release_targets.add(d)
        division_local={};division_support={}
        if cfg.get('division_local_assignment',False):
            for s,pairs in fork_source.items():
                old=tuple(sorted(out.get(s,[])))
                if len(old)>1:continue
                for a,b,p in pairs:
                    if p<cfg['fork_threshold'] or min(eq.get((s,a),0),eq.get((s,b),0))<cfg['new_fork_min_edge']:continue
                    owners={prev[d] for d in (a,b) if d in prev and prev[d]!=s}
                    if any(len(out.get(owner,[]))!=1 for owner in owners):continue
                    event=(s,tuple(sorted((a,b))))
                    division_local.setdefault(s,set()).add(event[1])
                    for owner in owners:division_support.setdefault(owner,set()).add(event)
        selected=[];fallback=0;newforks=0;removedforks=0
        for t,srcs in sorted(frames.items()):
            targets=frames.get(t+1,[])
            if not targets:continue
            # One event per source, one parent per target. Events may have zero, one, or two children.
            events=[];values=[];src_index={s:i for i,s in enumerate(srcs)};target_index={d:i+len(srcs) for i,d in enumerate(targets)}
            for s in srcs:
                old=tuple(sorted(out.get(s,[])));h=chain(s,prev,pos);v=(h[0]-h[-1])/max(1,len(h)-1)
                options=[()]+[(d,) for d in bysource.get(s,[])]
                reject_old=len(old)==1 and cfg.get('edge_low_confidence_pruning',False) and eq_high.get((s,old[0]),1.)<=cfg.get('edge_pruning_max',.01)
                if len(old)==1 and cfg.get('edge_consensus_gate',False):
                    # Keep the incumbent unless every ensemble member rejects it
                    # and every member supports the replacement. This threshold is
                    # selected before the reserved whole-movie audit.
                    options=[ds for ds in options if ds and (ds==old or
                             (eq_high.get((s,old[0]),1.)<=cfg.get('edge_old_max',.5)
                              and eq_low.get((s,ds[0]),0.)>=cfg.get('edge_new_min',.995)))]
                    if old[0] in release_targets or reject_old:options.append(())
                recoverable=[]
                if not old and cfg['endpoint_recovery'] and len(h)>=cfg.get('endpoint_min_history',3):
                    recoverable=[d for d in bysource.get(s,[]) if (d not in prev or (s,d) in reassign) and len(chain(d,out,pos))>=cfg.get('endpoint_min_future',3) and eq[(s,d)]>=cfg['endpoint_probability'] and np.linalg.norm(pos[d]-pos[s]-v)<=cfg.get('endpoint_motion_error',4.5) and np.linalg.norm(pos[d]-pos[s])<=cfg.get('endpoint_max_distance',8.)]
                for a,b,p in fork_source.get(s,[]):
                    pair=tuple(sorted([a,b]));isold=pair==old
                    threshold=cfg.get('fork_repair_threshold',.1) if len(old)==2 and cfg.get('fork_repair',False) else cfg['fork_threshold']
                    if (isold and p>=cfg['fork_veto']) or (not isold and p>=threshold and min(eq.get((s,a),0),eq.get((s,b),0))>=cfg['new_fork_min_edge']):options.append(pair)
                if cfg['mode']=='fork_only' and len(old)==1:options=[ds for ds in options if ds==old or (len(ds)==2 and old[0] in ds) or (not ds and old[0] in release_targets)]
                if cfg['mode']=='fork_only' and not old:options=[ds for ds in options if not ds or (cfg.get('fork_from_endpoints',False) and len(ds)==2 and all(d not in prev for d in ds))]
                if len(old)==2 and cfg['preserve_inherited_divisions'] and fq.get((s,*old),1.)>=cfg['fork_veto']:
                    options=[ds for ds in options if len(ds)==2] if cfg.get('fork_repair',False) else [old]
                if s in protected:options=[old]
                if cfg.get('division_local_assignment',False):
                    if s in division_local:options=[ds for ds in options if ds==old or ds in division_local[s]]
                    elif s not in division_support:options=[old]
                    else:options=[ds for ds in options if len(ds)<=1]
                for ds in dict.fromkeys(options):
                    if not ds:
                        value=-float(cfg.get('disappearance_penalty',5.)) if old else (-1. if recoverable else 0.)
                        if old and (cfg.get('learned_disappearance',False) or reject_old or (len(old)==1 and old[0] in release_targets) or s in division_support):
                            continuation=max((eq.get((s,d),0.) for d in bysource.get(s,[])),default=0.)
                            value=cfg['edge_weight']*np.log(np.clip(1-continuation,1e-5,1.))
                    else:
                        if not old and len(ds)==1 and ds[0] not in recoverable:continue
                        value=sum(cfg['edge_weight']*(float(logit(np.clip(eq.get((s,d),.001),1e-5,1-1e-5))) if cfg['edge_logit'] else np.log(max(eq.get((s,d),.001),1e-6)))-.03*cfg['motion_weight']*np.linalg.norm(pos[d]-pos[s]-.5*v) for d in ds)
                        if len(ds)==2:
                            if cfg.get('event_normalize_children',False):value/=2.
                            p=fq.get((s,*ds),.001)
                            if len(old)==2 and cfg.get('fork_repair',False):value+=cfg['fork_weight']*np.log(max(p,1e-6))
                            else:
                                evidence=cfg['fork_weight']*(float(logit(np.clip(p,1e-5,1-1e-5)))-float(logit(np.clip(cfg['fork_threshold'],1e-5,1-1e-5))))
                                value+=max(0.,evidence) if ds==old else evidence
                        if ds!=old:value-=cfg['fork_change_penalty'] if len(ds)==2 or len(old)==2 else cfg['edge_change_penalty']
                        else:value+=.1
                    events.append((s,ds));values.append(value)
            if not events:continue
            rr=[];cc=[]
            for j,(s,ds) in enumerate(events):
                rr.append(src_index[s]);cc.append(j)
                for d in ds:rr.append(target_index[d]);cc.append(j)
            vv=list(np.ones(len(rr)));event_lookup={event:i for i,event in enumerate(events)};constraints=[]
            if cfg.get('division_local_assignment',False):
                for j,(s,ds) in enumerate(events):
                    if s not in division_support or ds==tuple(sorted(out.get(s,[]))) or len(ds)==2:continue
                    support=[event_lookup[e] for e in division_support[s] if e in event_lookup]
                    constraints.append((j,support))
                for k,(j,support) in enumerate(constraints):
                    row=len(srcs)+len(targets)+k;rr.append(row);cc.append(j);vv.append(1.)
                    for q in support:rr.append(row);cc.append(q);vv.append(-1.)
            mat=coo_matrix((vv,(rr,cc)),shape=(len(srcs)+len(targets)+len(constraints),len(events))).tocsc()
            lower=np.r_[np.ones(len(srcs)),np.zeros(len(targets))];upper=np.ones(len(srcs)+len(targets))
            if constraints:lower=np.r_[lower,np.full(len(constraints),-np.inf)];upper=np.r_[upper,np.zeros(len(constraints))]
            result=milp(-np.asarray(values),integrality=np.ones(len(events)),bounds=Bounds(0,1),constraints=LinearConstraint(mat,lower,upper),options={'time_limit':cfg['solver_seconds'],'mip_rel_gap':0.001})
            if result.x is None or not result.success:
                fallback+=1;selected.extend(dict(e) for (s,d),e in original.items() if int(nodes[s]['t'])==t);continue
            for use,(s,ds) in zip(result.x,events):
                if use<.5:continue
                old=out.get(s,[])
                if len(ds)==2 and set(ds)!=set(old):newforks+=1
                if len(old)==2 and len(ds)!=2:removedforks+=1
                for d in ds:selected.append(dict(original.get((s,d),{'source_id':s,'target_id':d,'edge_prob':eq.get((s,d),0.),'joint_event':1})))
        check_out,check_prev,_,_=structure(nodes,selected)
        assert len(check_prev)==len(selected) and all(len(v)<=2 for v in check_out.values())
        assert all(nodes[e['target_id']]['t']==nodes[e['source_id']]['t']+1 for e in selected)
        newset={(int(e['source_id']),int(e['target_id'])) for e in selected};oldset=set(original)
        stats={'joint_edges_added':len(newset-oldset),'joint_edges_removed':len(oldset-newset),'joint_divisions_added':newforks,'joint_divisions_removed':removedforks,'solver_fallback_frames':fallback,'seconds':round(time.time()-start,2)}
        if cfg.get('endpoint_reassignment',False):stats['endpoint_reassignments']=len(newset&reassign)
        if cfg.get('division_reassignment',False):stats['division_target_reassignments']=len(newset&division_reassign)
        print('JOINT_REFINEMENT',name,json.dumps(stats),flush=True)
        return nodes,selected,stats
