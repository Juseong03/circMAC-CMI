#!/bin/bash
#===============================================================================
# 2-GPU Experiment Distribution Guide
#
# 3 Servers × 2 GPUs = 6 slots → 180 total runs
#
# ┌──────────┬───────────────────────┬───────────────────────┐
# │  Server  │       GPU 0           │       GPU 1           │
# ├──────────┼───────────────────────┼───────────────────────┤
# │ Server 1 │ exp2_ss1  (36 runs)   │ exp2_ss5  (39 runs)   │
# │ Server 2 │ exp1_fair (27 runs)   │ exp1_max  (21 runs)   │
# │ Server 3 │ exp3+exp5 (24 runs)   │ exp4+exp6 (33 runs)   │
# └──────────┴───────────────────────┴───────────────────────┘
#
# Total: 36 + 39 + 27 + 21 + 24 + 33 = 180 runs
#
# NOTE: exp1 CircMAC-PT requires exp2 pretrained model (best.pt).
#       Run Server 1 (exp2) FIRST, then Server 2 (exp1).
#       Server 3 is independent and can run anytime.
#===============================================================================

cat << 'GUIDE'

============================================================
  How to run experiments on each server
============================================================

# Step 1: On ALL servers, pull latest code
git pull

# Step 2: Make scripts executable
chmod +x scripts/2gpu/*.sh

------------------------------------------------------------
# Server 1: Pretraining Strategy (exp2)
#   - GPU 0: ss1 data (36 runs)
#   - GPU 1: ss5 data (39 runs)
------------------------------------------------------------
nohup ./scripts/2gpu/exp2_ss1.sh 0 > logs/server1_gpu0.log 2>&1 &
nohup ./scripts/2gpu/exp2_ss5.sh 1 > logs/server1_gpu1.log 2>&1 &

# Monitor:
# tail -f logs/server1_gpu0.log
# tail -f logs/server1_gpu1.log

------------------------------------------------------------
# Server 2: Pretrained Model Comparison (exp1)
#   - GPU 0: Fair comparison, max_len=440 (27 runs)
#   - GPU 1: Max performance (21 runs)
#   NOTE: CircMAC-PT skipped if exp2 not done yet
------------------------------------------------------------
nohup ./scripts/2gpu/exp1_fair.sh 0 > logs/server2_gpu0.log 2>&1 &
nohup ./scripts/2gpu/exp1_max.sh  1 > logs/server2_gpu1.log 2>&1 &

------------------------------------------------------------
# Server 3: Architecture & Ablation (exp3+4+5+6)
#   - GPU 0: Encoder comparison + Interaction (24 runs)
#   - GPU 1: Ablation + Site head (33 runs)
------------------------------------------------------------
nohup ./scripts/2gpu/exp3_exp5.sh 0 > logs/server3_gpu0.log 2>&1 &
nohup ./scripts/2gpu/exp4_exp6.sh 1 > logs/server3_gpu1.log 2>&1 &

============================================================
  Check running experiments
============================================================
# See GPU usage
nvidia-smi

# Check running processes
ps aux | grep "python training.py\|python pretraining.py"

# Check progress (count completed experiments)
ls saved_models/circmac/ 2>/dev/null | wc -l

GUIDE
