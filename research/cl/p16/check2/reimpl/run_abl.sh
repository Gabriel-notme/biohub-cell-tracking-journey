#!/bin/bash
cd /workspace/cl
RULE_POOL=6 python3 rule_eval.py p15 ideas.chk2_reimpl_mine "$(cat p16/check2/reimpl/abl.json)" > p16/check2/reimpl/abl_rule_eval.out 2>&1
cp rule_eval_last_ideas_chk2_reimpl_mine.json p16/check2/reimpl/rows_abl.json
RULE_POOL=6 python3 rule_eval.py p15 ideas.chk2_reimpl_p19r "$(cat p16/check2/reimpl/grid_kw.json)" > p16/check2/reimpl/grid_rule_eval.out 2>&1
cp rule_eval_last_ideas_chk2_reimpl_p19r.json p16/check2/reimpl/rows_grid.json
echo ABL_FIN
