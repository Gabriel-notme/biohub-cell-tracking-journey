#!/bin/bash
cd /workspace/cl
SCRIPT=p_stage4.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p6_config.json hold36 p6n
SCRIPT=p_stage4.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p6_config.json prev4 p6n
PSTAGE_DUMP=/workspace/cl/cands_p8/audit32 SCRIPT=p_stage5.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p8_config.json audit32 p8
SCRIPT=p_stage4.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p6_config.json audit32 p6n
echo Q1_DONE
PSTAGE_DUMP=/workspace/cl/cands_p8/t127a SCRIPT=p_stage5.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p8_config.json t127a p8
PSTAGE_DUMP=/workspace/cl/cands_p8/t127b SCRIPT=p_stage5.py bash run_pstage.sh /workspace/p56stage /workspace/p56stage/p8_config.json t127b p8
echo Q1B_DONE
