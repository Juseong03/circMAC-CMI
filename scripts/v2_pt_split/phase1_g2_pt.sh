#!/bin/bash
#===============================================================================
# Phase 1 — GPU 2: (SKIP — mlm_cpcl_ssp already running externally)
#
# v2_ptm_mlm_cpcl_ssp PT is running via run_new_g0_mlm_cpcl_ssp.sh.
# No additional PT needed on this GPU for phase1.
#
# Once mlm_cpcl_ssp PT finishes, run phase2_g2_ft.sh for seed=3 FT.
#===============================================================================
echo "Phase 1 GPU 2: Nothing to do — v2_ptm_mlm_cpcl_ssp is already running."
echo "Wait for it to finish, then run: ./scripts/v2_pt_split/phase2_g2_ft.sh 2"
