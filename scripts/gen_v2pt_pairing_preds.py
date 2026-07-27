#!/usr/bin/env python3
"""
gen_v2pt_pairing_preds.py
=========================
v2_pt_pairing 모델의 test 예측을 재생성하여
supplementary figure scripts가 읽는 두 가지 경로에 저장한다.

저장 경로 (우선순위 순):
  1. logs/circmac/v2_pt_pairing_s{seed}/{seed}/best_preds/test_preds.pkl
     (tensor format: probs_sites, labels_sites, lengths_sites)
  2. eval_results/preds/v2_pt_pairing_s{seed}/test_preds.pkl
     (DataFrame format: sample_idx, position, label, prob)

Usage:
    python scripts/gen_v2pt_pairing_preds.py --device 0
    python scripts/gen_v2pt_pairing_preds.py --device 0 --seeds 1 2 3
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CircRNABindingSitesDataset
from trainer import Trainer
from utils import get_device
from utils_config import get_model_config

# ── Constants ──────────────────────────────────────────────────────────────────
VIZ_DIR   = ROOT / "models_for_viz"
LOGS_DIR  = ROOT / "logs"
PRED_DIR  = ROOT / "eval_results" / "preds"
DATA_TEST = ROOT / "data" / "df_test_final.pkl"

EXP_TPL    = "v2_pt_pairing"
MODEL_NAME = "circmac"
D_MODEL    = 128
N_LAYER    = 6
MAX_LEN    = 1022
BATCH_SIZE = 64
TASK       = "sites"


def _get_vocab_size(ckpt_path: Path) -> int:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k, v in state.items():
        if "embedding" in k and "weight" in k and hasattr(v, "ndim") and v.ndim == 2 and v.shape[-1] == D_MODEL:
            return int(v.shape[0])
    return 11


def run_seed(seed: int, device: torch.device) -> None:
    exp      = f"{EXP_TPL}_s{seed}"
    ckpt     = VIZ_DIR / MODEL_NAME / exp / str(seed) / "train" / "model.pth"
    logs_out = LOGS_DIR / MODEL_NAME / exp / str(seed) / "best_preds" / "test_preds.pkl"
    pred_out = PRED_DIR / f"{exp}" / "test_preds.pkl"

    if not ckpt.exists():
        print(f"[SKIP] {exp} — checkpoint not found: {ckpt}")
        return

    # Skip if logs pkl already correct (has probs_sites key)
    if logs_out.exists():
        with open(logs_out, "rb") as f:
            d = pickle.load(f)
        if "probs_sites" in d:
            print(f"[OK]   {exp} — logs pkl already exists, skipping")
            return

    print(f"\n[RUN]  {exp}")
    print(f"  ckpt: {ckpt}")

    # Load test data
    df_test = pd.read_pickle(DATA_TEST)
    ds = CircRNABindingSitesDataset(df_test, max_len=MAX_LEN + 2, target_type="mirna", k=1, k_target=1)

    # Build trainer
    vocab_size = _get_vocab_size(ckpt)
    cfg = get_model_config(model_name=MODEL_NAME, d_model=D_MODEL, n_layer=N_LAYER,
                           vocab_size=vocab_size, verbose=False)

    tr = Trainer(seed=seed, device=device, experiment_name=exp, verbose=False)
    tr.set_dataloader(ds, part=2, batch_size=BATCH_SIZE, num_workers=4)
    tr.define_model(cfg, model_name=MODEL_NAME, pretrain=False,
                    is_cross_attention=True, interaction="cross_attention",
                    use_unified_head=False, site_head_type="conv1d")
    tr.set_pretrained_target(target="mirna", rna_model="rnabert")
    tr.task = TASK
    tr.rc   = False
    tr.site_class_weights = None
    tr.alpha = 0.5
    tr.beta  = 0.5

    tr.load_model_from_path(str(ckpt), verbose=True)
    tr.model.eval()

    # Run evaluation
    _, tensors, _ = tr.step_loader(tr.test_loader, 0, is_train=False, data_type="Test")

    # ── Format 1: tensor format → logs/best_preds/ ──
    preds_raw  = tensors["preds_sites"]    # (N, L-1, 2) or (N, L-1)
    labels_raw = tensors["labels_sites"]   # (N, L) — includes CLS at position 0

    import torch.nn.functional as F
    if preds_raw.dim() == 3 and preds_raw.size(-1) == 2:
        probs = torch.softmax(preds_raw.float(), dim=-1)[..., 1]
    else:
        probs = torch.sigmoid(preds_raw.float().squeeze(-1))

    labels_no_cls = labels_raw[:, 1:]   # (N, L-1), CLS removed

    save_dict = {
        "probs_sites":   probs.cpu(),
        "labels_sites":  labels_no_cls.cpu(),
        "lengths_sites": tensors.get("lengths_sites", torch.zeros(probs.shape[0])).cpu(),
    }
    logs_out.parent.mkdir(parents=True, exist_ok=True)
    with open(logs_out, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"  → Saved logs pkl: {logs_out}")

    # Verify AUPRC
    from sklearn.metrics import average_precision_score
    lbl_flat = labels_no_cls.reshape(-1).numpy()
    prb_flat = probs.reshape(-1).numpy()
    valid    = lbl_flat != -100
    ap = average_precision_score(lbl_flat[valid].astype(int), prb_flat[valid].astype(float))
    print(f"  AUPRC (AP) = {ap:.4f}  (n_valid={valid.sum():,}  pos_rate={lbl_flat[valid].mean():.4f})")

    # ── Format 2: DataFrame format → eval_results/preds/ ──
    n_samples = probs.shape[0]
    rows = []
    for i in range(n_samples):
        L = int(labels_no_cls.shape[1])
        for pos in range(L):
            lab = int(labels_no_cls[i, pos].item())
            if lab == -100:
                break
            rows.append({"sample_idx": i, "position": pos,
                         "label": lab, "prob": float(probs[i, pos].item())})

    df_preds = pd.DataFrame(rows)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_pickle(pred_out)
    print(f"  → Saved eval pkl: {pred_out}  ({len(df_preds):,} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds",  type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Experiment: {EXP_TPL}  seeds={args.seeds}\n")

    for seed in args.seeds:
        run_seed(seed, device)

    print("\nDone. Now re-run supplementary figure scripts:")
    print("  python figures_paper/fig_roc_curves/fig_supp_pr_noise_pair.py")
    print("  python figures_paper/fig_roc_curves/fig_supp_pr_circmac.py")
    print("  python figures_paper/fig_roc_curves/fig_supp_label_noise.py")
    print("  python figures_paper/fig_roc_curves/fig_pr_from_preds.py")


if __name__ == "__main__":
    main()
