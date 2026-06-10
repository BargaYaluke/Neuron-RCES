#!/usr/bin/env bash
# =============================================================================
# 层级(池化范围)消融  (ResNet18 / CIFAR10)
#   自变量 = "在多大范围里凑预算 + 在什么尺度上比较"(池化范围 + 标准化方式)
#
#   论证目标(与方向性消融不同): §3.3 的设计声明 —— NRC 只在层内可比、跨层不可比,
#   因此选择必须逐层进行。该论证由两部分共同构成:
#     (A) global_raw 必须失败  => 证明"跨层不可比"是真问题(不是稻草人)
#     (B) global_z / global_rank 恢复 => 证明问题出在跨层"尺度"上, 而非全局选择本身
#   合起来: 机制(选最不critical的单元)在全局也成立, 唯一坏事的是未归一化的跨层尺度;
#   且逐层固定 k 仍以"简单 + 均匀覆盖"取胜。
#
#   四个臂(总预算 B 严格相等; 每层占用 c_l 是被测变量, 故意放开):
#     layerwise   : 逐层取最低 k 个 (= 主方法本身), 每层都取 => c_l 平线
#     global_raw  : 全网原始 NRC 入同一池, 取全局最低 B 个 (不做任何跨层校正)
#     global_z    : 先层内 z-score (减层均值/层标准差), 再入池取全局最低 B 个
#     global_rank : 先层内 rank/分位数 (r+0.5)/N_l, 再入池取全局最低 B 个
#
#   一次 attribution, 四个臂: NRC 分数只算一次并落盘, 四个臂全部从这同一份缓存派生掩码,
#   绝不重算 attribution。层内 z-score / rank 是在缓存的 l2-NRC 上做的额外变换, 不动分子。
#   与方向性消融"共享同一份缓存"(分数与张量形状都与 k 无关) => 若方向性消融已跑, 零额外成本。
#
#   微调流程克隆方向性消融: 干净 CE、gamma=1 不做 consolidation(只看未插值 final_result)、
#   epochs/lr/wd/bs/optim/seed 全部相同。robust acc 用 PGD-10。占用直方图在掩码生成时即落盘。
#
#   用法:  bash run_layer_ablation.sh
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 (改成与主表一致) -----------------------------
CKPT=/data/coding/RiFT/ResNet18_CIFAR10.pth   # 四臂同一个 robust 起点 (RobustBench RN18/CIFAR10)
MODEL=ResNet18
DATASET=CIFAR10
BUDGET_K=2            # 每层 k => 总预算 B=sum_l min(k,|S_l|)。务必与"主表"的 k 一致。
                     #   注意: 方向性消融脚本用的是 NEURONS=4。本脚本与缓存共享与 k 无关
                     #   (分数+形状都与 k 无关), 故这里可独立设 k; 但 layerwise 参照臂会用
                     #   此 k 重新微调。若你的主表用 k=4, 把这里也改成 4。
EPOCHS=30
LR=0.0003            # 小数形式, 必须与目录名一致 (不要写 3e-4)
WD=0.0005
BS=256
OPTIM=SGDM
NUM_GRAD_BATCHES=10  # 对抗梯度(分子)累积 batch 数 (仅在需要新算 attribution 时用到)
ADV_SUBSET=10000     # 固定对抗子集 B 大小 (仅新算 attribution 时)
ADV_SEED=0          # 子集采样种子 (与 attribution 一起被缓存固定)
SEED=0              # 训练种子 — 四臂共用 (微调随机性是共同项; 本消融选择全确定, 无 mask_seed)

# 缓存复用: 默认指向方向性消融的 SEL 目录 (分数+形状与 k 无关 => 任意 k 都可复用)。
# 若不存在则本脚本自己跑一次 --cal_neuron_mrc --norm_mode l2 到 _layer_sel_s${SEED}。
DIR_NEURONS=4        # 方向性消融跑选择时用的 NEURONS(只影响缓存目录名, 不影响分数/形状)

# layerwise 参照臂: 若你已有"主方法/lowest"在相同 k 下的微调结果, 填它的目录跳过重训;
# 留空则本脚本派生 layerwise 掩码并微调一遍(自包含)。
LAYERWISE_DIR=""
# ---------------------------------------------------------------------------

BASE=results/${MODEL}_${DATASET}/checkpoint
SUF=rift_${OPTIM}_lr=${LR}_wd=${WD}_epochs=${EPOCHS}_neurons=${BUDGET_K}

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint 不存在: $CKPT  (改脚本顶部的 CKPT)"; exit 1
fi

# common finetune step (四臂共用): $1 = run_tag (= 该臂目录名后缀, neurons=BUDGET_K 之后)
common_ft() {
  python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
    --neurons_per_layer $BUDGET_K --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS \
    --batch_size $BS --lr_scheduler cosine --momentum 0.9 --seed "$SEED" \
    --robust_eval_interval 1 --run_tag "$1"
}

# =============================================================================
# 第一步: 复用同一份 attribution 缓存 (与方向性消融共享)。
#   优先用方向性消融的 _dir_sel_s${SEED}; 不存在则本脚本算一次 l2-NRC 到 _layer_sel_s${SEED}。
#   分数(neuron_mrc_list.npy)与形状(neuron_masks.pth)都与 k 无关, 故 DIR_NEURONS != BUDGET_K
#   也能复用。
# =============================================================================
DIR_SUF=rift_${OPTIM}_lr=${LR}_wd=${WD}_epochs=${EPOCHS}_neurons=${DIR_NEURONS}
DIR_SEL=$BASE/${DIR_SUF}_dir_sel_s${SEED}
LAYER_SEL=$BASE/${SUF}_layer_sel_s${SEED}

if [ -f "$DIR_SEL/neuron_mrc_list.npy" ] && [ -f "$DIR_SEL/neuron_masks.pth" ]; then
  SEL=$DIR_SEL
  echo "==================== 第一步: 复用方向性消融缓存 ===================="
  echo "  复用: $SEL  (零额外 attribution 成本)"
else
  SEL=$LAYER_SEL
  echo "==================== 第一步: 新算一次 attribution (缓存) ===================="
  echo "  方向性缓存缺失, 本脚本自算 -> $SEL"
  python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
    --cal_neuron_mrc --norm_mode l2 \
    --neurons_per_layer $BUDGET_K --num_grad_batches $NUM_GRAD_BATCHES \
    --adv_subset $ADV_SUBSET --adv_seed $ADV_SEED --seed $SEED \
    --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS --run_tag layer_sel_s${SEED}
fi

SCORES=$SEL/neuron_mrc_list.npy
TEMPLATE=$SEL/neuron_masks.pth
for f in "$SCORES" "$TEMPLATE"; do
  [ -f "$f" ] || { echo "ERROR: 缺少缓存产物 $f"; exit 1; }
done

# =============================================================================
# 第二步 + 第三步: 从同一份缓存派生每个臂的掩码 (并即刻落盘 occupancy.json), 再用克隆协议微调。
# =============================================================================
declare -a CELLS=()   # 收集 (label=DIR) 供汇总用

derive_and_ft() {   # $1 = arm (= label = run_tag 主体)
  local arm="$1"
  local tag=layer_${arm}_s${SEED}
  local dir=$BASE/${SUF}_${tag}
  echo ">>> [派生] arm=$arm  -> $dir"
  python derive_layer_mask.py --scores "$SCORES" --template "$TEMPLATE" \
    --arm "$arm" --budget_k $BUDGET_K --out "$dir/neuron_masks.pth"
  echo ">>> [微调] arm=$arm  tag=$tag"
  common_ft "$tag"
  CELLS+=("${arm}=${dir}")
}

# layerwise 参照臂: 复用已有目录或自派生+微调
if [ -n "$LAYERWISE_DIR" ] && [ -f "$LAYERWISE_DIR/neuron_masks.pth" ]; then
  echo ">>> [layerwise] 复用已有微调目录: $LAYERWISE_DIR"
  # 即便复用, 也补一份 occupancy.json 以便画图/汇总 (掩码已存在, 不重新选择)
  python derive_layer_mask.py --scores "$SCORES" --template "$TEMPLATE" \
    --arm layerwise --budget_k $BUDGET_K \
    --out "$LAYERWISE_DIR/neuron_masks.layerwise_ref.pth" \
    --occ_out "$LAYERWISE_DIR/occupancy.json" >/dev/null
  CELLS+=("layerwise=${LAYERWISE_DIR}")
else
  derive_and_ft layerwise
fi

# 三个全局臂
derive_and_ft global_raw
derive_and_ft global_z
derive_and_ft global_rank

# =============================================================================
# 汇总: 等量(总)预算检查 + 占用(coverage / max_share) + clean/robust (参考 = layerwise)
# =============================================================================
echo "==================== 汇总 ===================="
CELL_ARGS=()
for c in "${CELLS[@]}"; do CELL_ARGS+=(--cell "$c"); done

python layer_metrics.py "${CELL_ARGS[@]}" --ref layerwise \
  --out $BASE/layer_ablation_table_s${SEED}.json \
  | tee $BASE/layer_ablation_table_s${SEED}.txt

echo "全部完成。"
echo "  每个臂最终精度(gamma=1, 未插值): <dir>/experiment_record.json -> final_result.{final_test_acc,final_robust_acc}"
echo "  每个臂占用(掩码生成时落盘): <dir>/occupancy.json -> layers[].c_l + coverage + max_share"
echo "  汇总表:   $BASE/layer_ablation_table_s${SEED}.txt"
