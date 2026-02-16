# Rerun Scripts

실패/누락된 실험만 재실행하는 스크립트입니다.
기존 스크립트를 중단할 필요 없이, 빈 GPU에서 별도로 실행하면 됩니다.
이미 완료된 실험은 자동으로 스킵합니다.

## 사용법

각 서버에서 `git pull` 후 실행:

### Server 2 (Exp2 finetune 재실행)
```bash
# GPU 0: ss1 finetune only (최대 27 runs, 완료된 것 자동 스킵)
./scripts/rerun/exp2_finetune_ss1.sh 0

# GPU 1: ss5 finetune only (최대 27 runs, 완료된 것 자동 스킵)
./scripts/rerun/exp2_finetune_ss5.sh 1
```

### Server 3 (Exp3/Exp4 누락분 재실행)
```bash
# GPU 0: Transformer(batch64) + CircMAC s3 (4 runs)
./scripts/rerun/exp3_missing.sh 0

# GPU 1: 누락된 ablation 5개 (15 runs)
./scripts/rerun/exp4_missing.sh 1
```

## 빠진 실험 요약

| Script | 내용 | Runs |
|--------|------|------|
| `exp2_finetune_ss1.sh` | ss1 pretrain → finetune (경로 버그로 스킵됨) | ~22 |
| `exp2_finetune_ss5.sh` | ss5 pretrain → finetune (경로 버그로 스킵됨) | ~27 |
| `exp3_missing.sh` | Transformer OOM(→bs64) + CircMAC s3 | 4 |
| `exp4_missing.sh` | no_circular_bias/pad, attn/mamba/cnn_only | 15 |

## 주의사항
- 기존 진행 중인 스크립트와 **같은 GPU를 사용하지 마세요**
- Exp2 finetune은 pretrain 모델이 있는 서버에서만 실행 가능
- 스크립트가 `[DONE]`으로 표시하면 이미 완료된 실험입니다
