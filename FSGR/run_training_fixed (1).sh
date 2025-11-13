#!/usr/bin/env bash
set -euo pipefail

# ============== 基础路径（按你当前项目结构已对齐） ==============
DATA_ROOT="./datasets/coco/images"                       # 包含 train2014/ 与 val2014/
ANN_DIR="./m2_annotations"
CLIP_PT="./.cache/clip/ViT-B-16.pt"                      # CLIP ViT-B/16 权重
TEXT_EMB="./text_embeddings/ram_ViT16_clip_text.pth"     # 文本嵌入
EXP_NAME="fsgr_fix"
BATCH=32
WORKERS=4

# ============== 可选：快速评估/限步（跑通/快验用） ==============
# 取消下面注释可启用“快速评估与限步”
# export FSGR_FAST_EVAL=1
# export FSGR_TRAIN_STEPS=200
# export FSGR_VAL_STEPS=80
# export FSGR_EVAL_STEPS=80
# 关闭 Meteor/Spice（避免 Java 依赖）
unset FSGR_USE_METEOR || true
unset FSGR_USE_SPICE || true

# ============== 路径检查/兜底 ==============
echo "[Check] 路径检查..."
if [[ ! -d "$DATA_ROOT/train2014" || ! -d "$DATA_ROOT/val2014" ]]; then
  echo "✗ 未找到 train2014/ 或 val2014/ 目录: $DATA_ROOT"
  exit 1
fi
[[ -f "$CLIP_PT" ]] || { echo "✗ 未找到 CLIP 权重: $CLIP_PT"; exit 1; }
[[ -f "$TEXT_EMB" ]] || { echo "✗ 未找到文本嵌入: $TEXT_EMB"; exit 1; }
[[ -d "$ANN_DIR" ]] || { echo "✗ 未找到标注目录: $ANN_DIR"; exit 1; }

echo "[Run] 使用参数："
echo "      DATA_ROOT       = $DATA_ROOT"
echo "      ANN_DIR         = $ANN_DIR"
echo "      CLIP_PT         = $CLIP_PT"
echo "      TEXT_EMB        = $TEXT_EMB"
echo "      BATCH_SIZE      = $BATCH"
echo "      WORKERS         = $WORKERS"
echo "      EXP_NAME        = $EXP_NAME"

# ============== 启动训练 ==============
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 可通过环境变量 EPOCHS 覆盖（默认 3 轮用于快速确认）
: "${EPOCHS:=3}"

python -u train_transformer.py \
  --exp_name "$EXP_NAME" \
  --batch_size "$BATCH" \
  --workers "$WORKERS" \
  --features_path "$DATA_ROOT" \
  --annotation_folder "$ANN_DIR" \
  --pre_vs_path "$CLIP_PT" \
  --text_embed_path "$TEXT_EMB" \
  --pre_name "ViT-B/16" \
  --xe_base_lr 2e-4 \
  --rl_base_lr 1e-5 \
  --epochs "$EPOCHS"
