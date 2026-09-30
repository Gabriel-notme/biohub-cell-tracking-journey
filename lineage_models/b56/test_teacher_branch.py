from teacher_branch import merge_teachers
import json
from pathlib import Path
def edge(s,t):return {'source_id':s,'target_id':t}
n={1:{'t':0},2:{'t':1},3:{'t':1},4:{'t':0},5:{'t':1}}
teacher=[edge(1,2),edge(1,3)];base=[edge(1,2),edge(4,3)]
nn,ee,st=merge_teachers(n,teacher,n,base)
assert {(e['source_id'],e['target_id']) for e in ee}=={(1,2),(1,3)} and st['teacher_divisions_added']==1
protected=base+[edge(4,5)]
assert merge_teachers(n,teacher,n,protected)[1]==protected
disagree=[edge(1,5),edge(4,3)]
assert merge_teachers(n,teacher,n,disagree)[1]==disagree
m={**n,9:{'t':2}}
assert merge_teachers(m,teacher,n,base)[2]['teacher_alignment_fallback']==1
assert merge_teachers(n,base,n,base)[1] is base
report={'compatible_fork_added':True,'incoming_conflict_resolved':True,'visual_divisions_preserved':True,'disagreeing_parent_retained':True,'unaligned_nodes_fall_back_to_B4':True,'unchanged_edge_order_preserved':True}
Path('teacher_branch_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
