#!/usr/bin/env bash
# =============================================================================
# gamma-consolidation 消融  (ResNet18 / CIFAR10)  —— 兑现 3.4 节的写作承诺
#   Eq.(8):  theta = theta_init + gamma*(theta_tuned - theta_init) ⊙ (1 - M̃)
#
# 这不是性质检验，而是防御性曲线：把主方法的增益归属于 NRC 选择本身，
# 证明 consolidation 只是一个可选后处理旋钮，不是性能来源。
#   * gamma=1（不做任何 consolidation）必须 == 主表里的 Neuron-RCES 数字
#   * 整条曲线上没有某个 gamma<1 的点好到让人觉得主方法藏了私货
#
# 纯推理扫描，不训练。它直接读主表微调每个 seed 留下的产物：
#   <cell>/init_params.pth      theta_initial（AT 起点，微调前 state_dict）
#   <cell>/best_params.pth      theta_tuned（主表模型，["model"]）
#   <cell>/neuron_masks.pth     M̃（落盘掩码，1=冻结，0=可训练）
#   <cell>/experiment_record.json  主表数字（gamma=1 免费端点校验用）
# 这些正是 main.py 每次微调都会写出的文件——所以本脚本通常无需重训，
# 只要把扫描指向主表那 3 个 seed 的 cell 目录即可。
#
# 用法:  bash run_gamma_ablation.sh
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 (改成与主表一致) -----------------------------
MODEL=ResNet18
DATASET=CIFAR10
NUM_CLASSES=10
INPUT_SIZE=32
NEURONS=4              # 每层选 k 个最不鲁棒神经元（=主表预算）
EPOCHS=30
LR=0.0003             # 小数形式，必须与目录名一致（不要写 3e-4）
WD=0.0005
BS=256
OPTIM=SGDM
SEEDS="0 1 2"         # 主表的 3 个 seed；单点冒烟测试用 "0"
RUN_TAG_PREFIX=""     # 若主表 cell 带 run_tag（如 a_nrc_ce），在此填前缀，下面拼出目录
BN_BUFFERS=tuned      # detail (c): tuned(默认,gamma=1精确对齐主表) | interp(两端都精确) | init
# ---------------------------------------------------------------------------

BASE=results/${MODEL}_${DATASET}/checkpoint
SUF=rift_${OPTIM}_lr=${LR}_wd=${WD}_epochs=${EPOCHS}_neurons=${NEURONS}

# 拼出每个 seed 的 cell 目录。若主表用了 run_tag，cell 名是 ${SUF}_${TAG}；
# 这里默认主表 cell 形如 ${SUF}_<prefix>s<seed>（与 run_norm/dir 的命名一致）。
CELL_ARGS=()
for S in $SEEDS; do
  if [ -n "$RUN_TAG_PREFIX" ]; then
    CELL=$BASE/${SUF}_${RUN_TAG_PREFIX}s${S}
  else
    CELL=$BASE/${SUF}
  fi
  if [ ! -d "$CELL" ]; then
    echo "ERROR: 找不到主表 cell 目录: $CELL"
    echo "  先用主表超参跑完 main.py 微调（产出 init/best/neuron_masks），或改 SUF/RUN_TAG_PREFIX。"
    exit 1
  fi
  for f in init_params.pth best_params.pth neuron_masks.pth; do
    [ -f "$CELL/$f" ] || { echo "ERROR: $CELL/$f 缺失（微调未完成？）"; exit 1; }
  done
  CELL_ARGS+=(--cell "$CELL")
done

echo ">>> gamma 扫描: ${SEEDS} (bn_buffers=$BN_BUFFERS)"
python gamma_consolidation.py \
  "${CELL_ARGS[@]}" \
  --model $MODEL --dataset $DATASET --num_classes $NUM_CLASSES --input_size $INPUT_SIZE \
  --batch_size $BS --bn_buffers $BN_BUFFERS \
  --out $BASE/gamma_consolidation_${BN_BUFFERS}.json \
  | tee $BASE/gamma_consolidation_${BN_BUFFERS}.txt

echo "完成。曲线数据: $BASE/gamma_consolidation_${BN_BUFFERS}.json"
echo "  gamma=1 行应与各 cell/experiment_record.json -> final_result 对齐（脚本已打印 Δ）。"
echo "  快速冒烟: 在 python 调用上加 --gammas 0,0.5,1 --no-ood。"
