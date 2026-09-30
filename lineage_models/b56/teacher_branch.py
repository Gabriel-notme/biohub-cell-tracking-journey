"""Retain visual-teacher links and add compatible motion-teacher divisions."""
from pathlib import Path
from collections import defaultdict
import copy,time

def merge_teachers(n3,e3,n4,e4):
    stats={'teacher_alignment_fallback':0,'teacher_divisions_added':0,'teacher_edges_removed':0}
    if n3!=n4:
        stats['teacher_alignment_fallback']=1
        return n4,e4,stats
    out3=defaultdict(set);out4=defaultdict(set);incoming={}
    for e in e3:out3[e['source_id']].add(e['target_id'])
    for e in e4:out4[e['source_id']].add(e['target_id']);incoming[e['target_id']]=e['source_id']
    pairs={(e['source_id'],e['target_id']) for e in e4};added=set();removed=set();claimed=set()
    for parent,children in sorted(out3.items()):
        if len(children)!=2 or len(out4[parent])!=1 or not out4[parent].issubset(children):continue
        if children&claimed:continue
        # Preserve every visual-teacher division, including competing parents.
        if any(c in incoming and incoming[c]!=parent and len(out4[incoming[c]])==2 for c in children):continue
        if any(c not in n4 or n4[c]['t']!=n4[parent]['t']+1 for c in children):continue
        for child in children:
            if child in incoming and incoming[child]!=parent:removed.add((incoming[child],child))
            if (parent,child) not in pairs:added.add((parent,child))
        claimed.update(children);stats['teacher_divisions_added']+=1
    if not added:return n4,e4,stats
    result=[e for e in e4 if (e['source_id'],e['target_id']) not in removed]+[{'source_id':s,'target_id':d} for s,d in sorted(added)]
    out=defaultdict(set);inc=defaultdict(set)
    for e in result:out[e['source_id']].add(e['target_id']);inc[e['target_id']].add(e['source_id'])
    assert all(len(x)<=2 for x in out.values()) and all(len(x)<=1 for x in inc.values())
    assert all(out[s]==cs for s,cs in out4.items() if len(cs)==2)
    stats['teacher_edges_removed']=len(removed);return n4,result,stats

class TeacherBranchRefiner:
    def __init__(self,paths,root,config=None,cache_dir=None):
        from motion_refine import MotionRefiner
        from visual_edge_refine import VisualEdgeRefiner
        from verified_recovery import VerifiedRecoveryRefiner
        from centroid_refine import CentroidRefiner
        self.paths=list(map(Path,paths));self.config=config;assert len(paths)==7
        self.motion=MotionRefiner(paths[:1],root,config['motion'],cache_dir)
        self.visual=VisualEdgeRefiner(paths[1:4],root,config['visual'],cache_dir)
        self.recovery=VerifiedRecoveryRefiner(paths[4:6],root,config['recovery'],cache_dir)
        self.centroid=CentroidRefiner(paths[6:],root,config['centroid'],cache_dir)
    def refine(self,name,nodes,edges):
        started=time.time();branches=[];fallback=0
        for reviewer in [self.motion,self.visual]:
            nn,ee,st=reviewer.refine(name,copy.deepcopy(nodes),copy.deepcopy(edges));fallback+=st.get('solver_fallback_frames',0)
            nn,ee,st=self.recovery.refine(name,nn,ee);fallback+=st.get('solver_fallback_frames',0)
            nn,ee,st=self.centroid.refine(name,nn,ee);branches.append((nn,ee))
        nn,ee,st=merge_teachers(*branches[0],*branches[1]);st.update(seconds=time.time()-started,solver_fallback_frames=fallback);return nn,ee,st
