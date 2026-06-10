#!/usr/bin/env bash
# =============================================================================
# 方向性消融  (ResNet18 / CIFAR10)  —— 唯一自变量 = "选择规则"(选哪 k 个神经元)
#
#   NRC = ||grad||_2 / (||W||_2 + eps)   在 AT 起点上只算 *一次*，逐层分数落盘缓存；
#   三个臂全部从这 *同一份* 缓存派生掩码，绝不重算 attribution：
#     lowest  : 每层 NRC 升序前 k 个 (= 主方法本身)
#     highest : 每层 NRC 降序前 k 个 (刻意训练我们声称最 critical 的单元)
#     random  : 每层在同一 alive 候选集中无放回均匀抽 k 个 (独立 mask_seed)
#
#   三臂每层可训练神经元数严格相等(预算从主方法自己的输出读回 => 逐神经元一致)，
#   分类头/BN/N_l<k 三条边界约定由缓存+模板自动继承(见 DIR_ABLATION.md)。
#   微调流程克隆主方法: 干净 CE、gamma=1 不做 consolidation(只看未插值的 final_result)、
#   epochs/lr/wd/bs/optim/seed 全部相同。robust acc 用 PGD-10(与主表一致)。
#
#   用法:  bash run_dir_ablation.sh
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 (改成与主表一致) -----------------------------
CKPT=/data/coding/RiFT/ResNet18_CIFAR10.pth   # 三臂同一个 robust 起点 (RobustBench RN18/CIFAR10)
MODEL=ResNet18
DATASET=CIFAR10
NEURONS=4              # 每层 k (= 主表预算，三臂相同)
EPOCHS=30
LR=0.0003             # 小数形式，必须与目录名一致 (不要写 3e-4)
WD=0.0005
BS=256
OPTIM=SGDM
NUM_GRAD_BATCHES=10   # 对抗梯度(分子)累积 batch 数
ADV_SUBSET=10000      # 固定对抗子集 B 大小
ADV_SEED=0           # 子集 B 的采样种子 (与 attribution 一起被缓存固定)
SEED=0               # 训练种子 — 三臂共用同一个 (微调随机性是共同项，不是自变量)
MASK_SEEDS="0 1 2"   # random 臂的独立 mask 种子 (展示随机臂的分布; lowest/highest 无种子)
RANDOM_POOL=valid    # random 抽样候选集: valid(=alive，与 lowest/highest 同集) | all
# ---------------------------------------------------------------------------

BASE=results/${MODEL}_${DATASET}/checkpoint
SUF=rift_${OPTIM}_lr=${LR}_wd=${WD}_epochs=${EPOCHS}_neurons=${NEURONS}

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint 不存在: $CKPT  (改脚本顶部的 CKPT)"; exit 1
fi

common_ft() {   # 三臂共用的微调步: $1 = run_tag (= 该臂目录名后缀)
  python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
    --neurons_per_layer $NEURONS --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS \
    --batch_size $BS --lr_scheduler cosine --momentum 0.9 --seed "$SEED" \
    --robust_eval_interval 1 --run_tag "$1"
}

# =============================================================================
# 第一步: 在 AT 起点上 *只算一次* NRC (纯 L_adv 梯度, contrastive 关, l2 归一化),
#         逐层分数缓存到 SEL 目录的 neuron_mrc_list.npy; 同目录 neuron_masks.pth
#         即"主方法的 lowest 掩码"，本消融把它当作 形状+预算 模板复用。
# =============================================================================
SEL_TAG=dir_sel_s${SEED}
SEL=$BASE/${SUF}_${SEL_TAG}
echo "==================== 第一步: 单次 attribution (缓存) ===================="
python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
  --cal_neuron_mrc --norm_mode l2 \
  --neurons_per_layer $NEURONS --num_grad_batches $NUM_GRAD_BATCHES \
  --adv_subset $ADV_SUBSET --adv_seed $ADV_SEED --seed $SEED \
  --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS --run_tag $SEL_TAG

SCORES=$SEL/neuron_mrc_list.npy
TEMPLATE=$SEL/neuron_masks.pth
for f in "$SCORES" "$TEMPLATE"; do
  [ -f "$f" ] || { echo "ERROR: 缺少缓存产物 $f (第一步是否成功?)"; exit 1; }
done

# =============================================================================
# 第二步 + 第三步: 从同一份缓存派生每个臂的掩码, 然后用克隆的协议微调。
# =============================================================================
declare -a CELLS=()   # 收集 (label=DIR) 供汇总用

derive_and_ft() {   # $1 = arm, $2 = run_tag, $3 = label, [$4 = mask_seed]
  local arm="$1" tag="$2" label="$3" mseed="${4:-0}"
  local dir=$BASE/${SUF}_${tag}
  echo ">>> [派生] arm=$arm  -> $dir"
  python derive_mask.py --scores "$SCORES" --template "$TEMPLATE" \
    --arm "$arm" --mask_seed "$mseed" --random_pool "$RANDOM_POOL" \
    --k_fallback $NEURONS --out "$dir/neuron_masks.pth"
  echo ">>> [微调] arm=$arm  tag=$tag"
  common_ft "$tag"
  CELLS+=("${label}=${dir}")
}

# lowest (= 主方法; 应与模板 Jaccard=1) 与 highest 各一次
derive_and_ft lowest  dir_lowest_s${SEED}  lowest
derive_and_ft highest dir_highest_s${SEED} highest

# random 臂: 多个独立 mask 种子, 展示随机选择的分布
for M in $MASK_SEEDS; do
  derive_and_ft random dir_rand_m${M}_s${SEED} "random_m${M}" "$M"
done

# =============================================================================
# 汇总: 等量约束检查 + 臂间 Jaccard + clean/robust (参考 = lowest)
# =============================================================================
echo "==================== 汇总 ===================="
CELL_ARGS=()
for c in "${CELLS[@]}"; do CELL_ARGS+=(--cell "$c"); done
python dir_metrics.py "${CELL_ARGS[@]}" --ref lowest \
  --out $BASE/dir_ablation_table_s${SEED}.json \
  | tee $BASE/dir_ablation_table_s${SEED}.txt

echo "全部完成。"
echo "  每个臂最终精度(gamma=1, 未插值): <dir>/experiment_record.json -> final_result.{final_test_acc,final_robust_acc}"
echo "  汇总表: $BASE/dir_ablation_table_s${SEED}.txt"
