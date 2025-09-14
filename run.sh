#!/usr/bin/env bash
set -euo pipefail

# ==== 路徑參數（依你的環境改）====
ROOT_PRICE=/home/sally/dataset/data_preprocessing/price_data
ROOT_FIN=/home/sally/dataset/data_preprocessing/financial
ROOT_NEWS=/home/sally/dataset/data_preprocessing/news/monthly_embeddings
ROOT_EVENT=/home/sally/dataset/data_preprocessing/event_type_PCA
ROOT_LABEL=/home/sally/dataset/data_preprocessing/esg_label/esg_npy
ROOT_SYM=/home/sally/dataset/ticker/nyse/yearly_symbol

# ==== 訓練切分 ====
TRAIN_YEARS=2015-2019
VAL_YEARS=2020-2021
TEST_YEARS=2022-2024

# ==== 模型/訓練參數（可自行調整）====
BATCH=32
HIDDEN=64
SLOTS=10
DROP=0.1
LR=3e-4
EPOCHS=100
NUM_WORKERS=4
PIN=""         # 想關掉就改成 ""
AMP="--amp"                # 想關掉就改成 ""
BOUND="--bounded_output"   # 想關掉就改成 ""
USE_REGS=""                # 想開正則就改成 "--use_regs"
L_INTRA=0.0                # slot 去相關
G_ATTN=0.0                 # 注意力熵
T_TEMP=0.0                 # 時間平滑
SAVE_DIR=./checkpoints_single
LOG_PREFIX=single_task

mkdir -p "$SAVE_DIR"

# 可選：不同任務用不同 seed（避免完全同初始化）
SEED_E=42
SEED_S=43
SEED_G=44

for TASK in E S G; do
  case "$TASK" in
    E) SEED=$SEED_E ;;
    S) SEED=$SEED_S ;;
    G) SEED=$SEED_G ;;
  esac

  echo "==================== Training TASK=$TASK ===================="
  python train.py \
    --root_price "$ROOT_PRICE" \
    --root_finance "$ROOT_FIN" \
    --root_news "$ROOT_NEWS" \
    --root_event "$ROOT_EVENT" \
    --root_label "$ROOT_LABEL" \
    --root_year_symbols "$ROOT_SYM" \
    --train_years "$TRAIN_YEARS" \
    --val_years "$VAL_YEARS" \
    --test_years "$TEST_YEARS" \
    --task "$TASK" \
    --batch_size $BATCH \
    --num_workers $NUM_WORKERS \
    $PIN \
    --hidden_dim $HIDDEN \
    --num_slots $SLOTS \
    --dropout $DROP \
    $BOUND \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay 1e-4 \
    $AMP \
    --grad_clip 1.0 \
    --seed $SEED \
    --lambd_intra $L_INTRA \
    --gamma_attn $G_ATTN \
    --tau_temp $T_TEMP \
    $USE_REGS \
    --save_dir "$SAVE_DIR" \
    --log_prefix "${LOG_PREFIX}_${TASK}"
done

echo "=== All tasks (E, S, G) finished. Check outputs under $SAVE_DIR ==="
