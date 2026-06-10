#!/usr/bin/env bash
# =============================================================================
# 归一化消融  (ResNet18 / CIFAR10)  —— 只换 NRC 的“分母(normalizer)”
#   NRC = ||grad||_2 / (denom + eps)   分子(grad L2) / 预算 k / 微调流程全部固定
#     none : denom = 1          原始梯度范数（无归一化，对照基准——最重要的一行）
#     l1   : denom = ||W||_1
#     l2   : denom = ||W||_2     你的方法（参考行）
#     linf : denom = ||W||_inf
#   （fan-in / cohort-L∞ 在“逐层 top-k”下与 none 完全等价，已剔除——见 NORM_ABLATION.md）
#
# 每个 cell = 「选择(--cal_neuron_mrc --norm_mode X)」→「微调」两步，
# 二者必须用相同的路径参数(optim/lr/wd/epochs/neurons/run_tag)，微调才能从选择
# 写出的目录读到 mask。四行用同一个 robust 起点、同一批对抗梯度、同一个 k、同一套微调超参，
# 唯一变量就是 --norm_mode。robust acc 用 PGD-10（与主表一致，微调循环自动评测）。
#
# 用法:  bash run_norm_ablation.sh
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 (改成与主表一致) -----------------------------
CKPT=/data/coding/RiFT/ResNet18_CIFAR10.pth   # 四行同一个 robust 起点
MODEL=ResNet18
DATASET=CIFAR10
NEURONS=4              # 每层选 k 个最不鲁棒神经元（=主表预算，四行相同）
EPOCHS=30
LR=0.0003             # 小数形式，必须与目录名一致（不要写 3e-4）
WD=0.0005
BS=256
OPTIM=SGDM
NUM_GRAD_BATCHES=10   # 对抗梯度(分子)累积的 batch 数——四行相同 => 分子 G 完全相同
ADV_SUBSET=10000
SEEDS="0"            # 只跑最佳点用 "0"; 要 mean±std 改成 "0 1 2"
NORMS="none l1 l2 linf"
# ---------------------------------------------------------------------------

BASE=results/${MODEL}_${DATASET}/checkpoint
SUF=rift_${OPTIM}_lr=${LR}_wd=${WD}_epochs=${EPOCHS}_neurons=${NEURONS}

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint 不存在: $CKPT  (改脚本顶部的 CKPT)"; exit 1
fi

common_ft() {   # 四行共用的微调步: $1 = run_tag, $2 = seed
  python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
    --neurons_per_layer $NEURONS --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS \
    --batch_size $BS --lr_scheduler cosine --momentum 0.9 --seed "$2" \
    --robust_eval_interval 1 --run_tag "$1"
}

for S in $SEEDS; do
  echo "==================== SEED $S ===================="
  for MODE in $NORMS; do
    TAG=norm_${MODE}_s${S}
    DIR=$BASE/${SUF}_${TAG}

    # ---------- 选择: 纯 L_adv 梯度 (contrastive 关) + 指定 normalizer ----------
    echo ">>> [seed $S] [$MODE] NRC 选择"
    python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
      --cal_neuron_mrc --norm_mode $MODE \
      --neurons_per_layer $NEURONS --num_grad_batches $NUM_GRAD_BATCHES \
      --adv_subset $ADV_SUBSET --adv_seed $S --seed $S \
      --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS --run_tag $TAG

    # ---------- 微调: 纯 CE、gradient-gating（四行完全一致） ----------
    echo ">>> [seed $S] [$MODE] 微调"
    common_ft $TAG $S
  done

  # ---------- 度量层 + 性能层汇总表 (参考 = l2) ----------
  echo ">>> [seed $S] 汇总: Jaccard / Kendall τ / clean / robust  (vs l2)"
  python norm_metrics.py \
    --cell none=$BASE/${SUF}_norm_none_s${S} \
    --cell l1=$BASE/${SUF}_norm_l1_s${S} \
    --cell l2=$BASE/${SUF}_norm_l2_s${S} \
    --cell linf=$BASE/${SUF}_norm_linf_s${S} \
    --ref l2 --out $BASE/norm_ablation_table_s${S}.json \
    | tee $BASE/norm_ablation_table_s${S}.txt
done

echo "全部完成。"
echo "  每个 cell 的最终精度: <dir>/experiment_record.json -> final_result.{final_test_acc,final_robust_acc}"
echo "  汇总表: $BASE/norm_ablation_table_s*.txt  (度量稳定性 + 性能稳定性, 参考行 = l2)"
