# Normalizer Ablation — what does dividing by `‖W‖₂` buy you?

**Setting:** ResNet-18 / CIFAR-10 only. One variable: the **denominator** of the
per-neuron NRC score. Everything else is held fixed.

```
NRC_{l,j} = ‖G_{l,j}‖₂ / (denom_{l,j} + ε_s)
            └── numerator: FIXED (L2 grad norm)   └── the only thing we vary
```

| row | name | `--norm_mode` | denominator | role |
|----:|------|---------------|-------------|------|
| 1 | No normalization | `none` | `1` | **the baseline** — answers "does normalizing help at all?" |
| 2 | L1 | `l1` | `‖W_{l,j}‖₁` | control |
| 3 | **L2 (ours)** | `l2` | `‖W_{l,j}‖₂` | **reference** |
| 4 | L∞ | `linf` | `‖W_{l,j}‖∞` | control |

## Why exactly these four (and not fan-in / cohort-L∞)

Selection is **per-layer top-k** (`main.py`: the k smallest NRC scores *within each
layer* are unfrozen). A normalizer that is **constant across the neurons of a layer**
cancels out of an in-layer `argsort` and therefore **cannot change the selection**:

> `argsort_j( gradⱼ / d_l ) = argsort_j( gradⱼ )`  when `d_l` is the same for every neuron `j` in layer `l`.

- **fan-in / parameter count** is identical for every output channel of a Conv
  (`C_in·kH·kW`) or every row of a Linear (`in_features`) → a per-layer constant →
  under per-layer top-k it gives **bit-identical selection to `none`** (Jaccard = τ = 1).
- **cohort-L∞ (RPF-style)** divides a whole cohort by one scalar (its max) → also a
  per-layer/per-block constant → **identical to `none`** under per-layer selection.

These two rows only *do* something when neurons from different layers compete for **one
global budget** (RPF ranks globally, not per-layer). That is a different experiment — it
relaxes the "k per layer" control — so it is intentionally **out of scope here**.
The four rows above all have **denominators that vary between neurons inside a layer**
(`L1/L2/L∞` of the weight differ per output channel), so each produces a genuinely
different selection. That is the comparison this table is designed to make.

## Controls (write this verbatim in the paper)

> *Holding the adversarial gradient, the budget, and the fine-tuning fixed, we vary
> only the normalizer.*

Fixed across all four rows:

- **backbone** ResNet-18; **dataset** CIFAR-10
- **adversarial gradient (numerator)** — same robust checkpoint, same PGD adv subset
  (`--adv_subset 10000 --adv_seed S`), same `--num_grad_batches 10`, **CE-only**
  (`--contrastive` off) → the numerator `‖G_{l,j}‖₂` is byte-for-byte identical
- **candidate set** — a neuron is eligible iff `‖W‖₂>0 ∧ ‖grad‖₂>0`; this test uses the
  L2 weight norm in *all* modes, so the set of selectable neurons is identical and only
  the *ranking* among them changes
- **budget** `k = NEURONS` per layer (default 4)
- **fine-tuning** clean CE, gradient-gating, cosine LR, same `lr/wd/epochs/optim/bs/seed`
- **attack** PGD-10, `eps=8/255` (`evaluate_cifar_robustness`, the main-table attack)

## Metrics reported (two layers)

**Selection stability** — relative to the L2 row:

- **Jaccard** of the top-k selected sets — *micro* (pooled over all neurons) and *macro*
  (mean over layers). High ⇒ the same neurons are chosen.
- **Kendall's τ** of the per-neuron NRC ranking — per-layer mean and global. High ⇒ the
  same ordering.
- Jaccard vs τ: τ can fall while Jaccard stays high if only the *bottom-k* is unchanged —
  which is all the method relies on, so report **both**.

**Result stability** — each row runs the full pipeline:

- **clean accuracy** ("std accuracy")
- **robust accuracy** (PGD-10)

## How to run

```bash
bash run_norm_ablation.sh        # edit CKPT + hyperparams at the top to match the main table
```

For each `MODE ∈ {none,l1,l2,linf}` it runs selection (`--cal_neuron_mrc --norm_mode MODE`)
→ fine-tune (gradient-gating), then `norm_metrics.py` prints the joint table. Set
`SEEDS="0 1 2"` for mean±std.

The metrics table alone (no re-train) can be regenerated any time:

```bash
python norm_metrics.py \
  --cell none=<dir> --cell l1=<dir> --cell l2=<dir> --cell linf=<dir> \
  --ref l2 --out norm_ablation_table.json
```

## Results table template

| Normalizer | Jaccard(sel) vs L2 (micro/macro) | Kendall τ vs L2 (layer/global) | Clean Acc % | Robust Acc % |
|---|---|---|---|---|
| None (`‖G‖₂`) | _ / _ | _ / _ | _ | _ |
| L1 | _ / _ | _ / _ | _ | _ |
| **L2 (ours)** | 1.000 / 1.000 | 1.000 / 1.000 | _ | _ |
| L∞ | _ / _ | _ / _ | _ | _ |

**How to read it.** The story you want is: (a) `none` selects a *different* set
(Jaccard < 1, τ < 1) **and** is *worse* on robust acc → normalization is necessary, not
cosmetic; (b) `l1`/`linf` land close to `l2` → the *fact* of normalizing matters more
than the exact p, and L2 is at least as good → your choice is principled, not arbitrary.
If instead every row ties `l2` on robust acc, the honest conclusion is "the normalizer
choice is second-order on ResNet-18/CIFAR-10," which still answers the reviewer.

## Artifacts per cell

`results/ResNet18_CIFAR10/checkpoint/rift_..._neurons=K_norm_<mode>_s<seed>/`

- `neuron_masks.pth` — selected sets + stamped `norm_mode` (authoritative; survives the
  fine-tune phase, which rewrites `experiment_record.json`)
- `neuron_mrc_list.npy` — per-neuron NRC scores (drives Kendall τ)
- `experiment_record.json` → `final_result.{final_test_acc, final_robust_acc}`
