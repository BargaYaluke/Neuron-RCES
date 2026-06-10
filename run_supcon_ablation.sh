#!/usr/bin/env bash
# =============================================================================
# SupCon 贡献拆分消融  (ResNet18 / CIFAR10)
#   (a) NRC 选择(纯 L_adv, --contrastive 关) + 纯 CE 微调   —— 主方法
#   (b) NRC 选择(L_adv + λ·SupCon, --contrastive 开) + 纯 CE 微调  —— supcon 变体
#   (c) 随机选同样数量神经元 + 纯 CE 微调                       —— 选择准则对照
#   + Jaccard 子表: (a) vs (b) 选中神经元重合率
#
# 每个 cell = 「产出 mask」→「微调」两步，二者必须用相同的路径参数
# (optim/lr/wd/epochs/neurons/run_tag)，微调才能从选择写出的目录读到 mask。
# 用法:  bash run_supcon_ablation.sh
# =============================================================================
set -euo pipefail

# ----------------------------- 配置 (按需修改) -----------------------------
CKPT=/data/coding/RiFT/ResNet18_CIFAR10.pth   # 三行同一个 robust 起点
MODEL=ResNet18
DATASET=CIFAR10
NEURONS=4
EPOCHS=30
LR=0.0003          # 注意: 用小数形式 0.0003 (不要 3e-4)，要和目录名一致
WD=0.0005
BS=256
OPTIM=SGDM
LAMBDA_CON=0.04
SEEDS="0"          # 只跑最佳点用 "0"; 要 mean±std 改成 "0 1 2"
# ---------------------------------------------------------------------------

BASE=results/${MODEL}_${DATASET}/checkpoint
SUF=rift_${OPTIM}_lr=${LR}_wd=${WD}_epochs=${EPOCHS}_neurons=${NEURONS}

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint 不存在: $CKPT  (改脚本顶部的 CKPT)"; exit 1
fi

common_ft() {   # 三行共用的微调步: $1 = run_tag
  python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
    --neurons_per_layer $NEURONS --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS \
    --batch_size $BS --lr_scheduler cosine --momentum 0.9 --seed "$2" \
    --robust_eval_interval 1 --run_tag "$1"
}

for S in $SEEDS; do
  echo "==================== SEED $S ===================="
  TAG_A=a_nrc_ce_s${S}
  TAG_B=b_nrc_supcon_s${S}
  TAG_C=c_rand_ce_s${S}
  A=$BASE/${SUF}_${TAG_A}
  B=$BASE/${SUF}_${TAG_B}
  C=$BASE/${SUF}_${TAG_C}

  # ---------- (a) NRC 选择: 纯 L_adv 梯度 (--contrastive 关) ----------
  echo ">>> [seed $S] (a) NRC/CE 选择"
  python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
    --cal_neuron_mrc --neurons_per_layer $NEURONS --num_grad_batches 10 \
    --adv_subset 10000 --adv_seed $S --seed $S \
    --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS --run_tag $TAG_A
  echo ">>> [seed $S] (a) 微调"
  common_ft $TAG_A $S

  # ---------- (b) NRC 选择: L_adv + λ·SupCon 梯度 (--contrastive 开) ----------
  echo ">>> [seed $S] (b) NRC/CE+SupCon 选择"
  python main.py --model $MODEL --dataset $DATASET --resume "$CKPT" \
    --cal_neuron_mrc --contrastive --lambda_ce 1.0 --lambda_con $LAMBDA_CON --temperature 0.07 \
    --neurons_per_layer $NEURONS --num_grad_batches 10 \
    --adv_subset 10000 --adv_seed $S --seed $S \
    --optim $OPTIM --lr $LR --wd $WD --epochs $EPOCHS --run_tag $TAG_B
  echo ">>> [seed $S] (b) 微调"
  common_ft $TAG_B $S

  # ---------- (c) 随机选同数量 (按 a 的 mask 逐层对齐) + 纯 CE 微调 ----------
  echo ">>> [seed $S] (c) 随机选择 (--like a)"
  python gen_mask.py --mode random --resume "$CKPT" --seed $S \
    --like $A/neuron_masks.pth --out $C/neuron_masks.pth
  echo ">>> [seed $S] (c) 微调"
  common_ft $TAG_C $S

  # ---------- Jaccard 子表: (a) vs (b) ----------
  echo ">>> [seed $S] Jaccard (a vs b)"
  python compare_masks.py --a $A/neuron_masks.pth --b $B/neuron_masks.pth | tee $B/jaccard_vs_a.txt

  echo "==== seed $S 完成 ===="
  echo "  (a) $A/experiment_record.json"
  echo "  (b) $B/experiment_record.json"
  echo "  (c) $C/experiment_record.json"
done

echo "全部完成。每个目录 experiment_record.json -> final_result.{final_test_acc,final_robust_acc}"
echo "归因: a-c = NRC 选择 vs 随机;  b-a = SupCon 对选择/鲁棒性的增量 (配 jaccard_vs_a.txt)"
