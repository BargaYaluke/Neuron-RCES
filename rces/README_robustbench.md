# Neuron-RCES Stage-1 — RobustBench backbones (WRN-28-10 + XCiT-S12)

A **standalone** pipeline (`main_robustbench.py` + the `rces/` package) that extends
Neuron-RCES from ResNet-18/34 to a conv backbone (**WRN-28-10**, CIFAR-10/100) and a
Transformer backbone (**robust XCiT-S12**, transferred to Tiny-ImageNet). It loads a
**public robust checkpoint** (no expensive adversarial pretraining) and runs cheap
gradient-gated **sparse adaptation** of the least-robust-critical neurons.

The original `main.py` is untouched.

## Install (server)

```bash
pip install git+https://github.com/RobustBench/robustbench.git   # pulls timm>=1.0.9 + autoattack
```

If `robustbench` cannot be installed, the XCiT path can fall back to a **clean
(non-robust)** timm XCiT via `--allow_timm_fallback` (only useful for plumbing tests;
not a robust starting point).

## The one rule that matters most

RobustBench models **normalize inputs internally**. This pipeline therefore keeps
inputs in `[0,1]` everywhere: dataloaders use `ToTensor` only (no `Normalize`), the
model is attacked **directly** (no `nn.Sequential(Normalize, model)`), and PGD
perturbs/clamps in `[0,1]`. Do **not** reuse the original `utils.py` PGD path here — it
double-normalizes for self-normalizing checkpoints.

## Step 0 — diagnostic first (recommended)

Print the module inventory + data-condition, then exit. **Send me this output** so I can
finalize qkv/head targeting against your exact `timm` build.

```bash
python main_robustbench.py --arch xcit-s12 --dataset tinyimagenet --diagnostic_only
python main_robustbench.py --arch wrn-28-10 --dataset cifar10    --diagnostic_only
```

## Main runs (the Neuron-RCES rows of the experiment matrix)

```bash
# WRN-28-10 on CIFAR-10  (eps=8/255, k=2 neurons/layer)
python main_robustbench.py --arch wrn-28-10 --dataset cifar10  --neurons_per_layer 2 --epochs 10

# WRN-28-10 on CIFAR-100
python main_robustbench.py --arch wrn-28-10 --dataset cifar100 --neurons_per_layer 2 --epochs 10

# robust XCiT-S12 -> Tiny-ImageNet (eps=4/255), attention-HEAD granularity
python main_robustbench.py --arch xcit-s12 --dataset tinyimagenet \
    --neurons_per_layer 2 --vit_neuron_granularity head \
    --head_warmup_epochs 2 --epochs 10

# same, UNIT granularity (each qkv/mlp output row is a neuron) — for the unit-vs-head ablation
python main_robustbench.py --arch xcit-s12 --dataset tinyimagenet \
    --neurons_per_layer 2 --vit_neuron_granularity unit \
    --head_warmup_epochs 2 --epochs 10
```

Add `--autoattack --autoattack_n 1000` for a RobustBench AutoAttack number on top of PGD.

## Key flags

| flag | meaning |
|------|---------|
| `--arch` | `wrn-28-10` \| `xcit-s12` |
| `--dataset` | `cifar10` \| `cifar100` \| `tinyimagenet` |
| `--model_name` | override the RobustBench checkpoint (defaults below) |
| `--neurons_per_layer` | k least-robust-critical neurons trainable per layer (default 2) |
| `--vit_neuron_granularity` | `unit` (Linear output row) \| `head` (attention head group on fused qkv) |
| `--num_heads` | force #heads if auto-detection fails (XCiT-S12 = 8) |
| `--per_proj_heads` | score Q/K/V heads separately (3·H candidates) instead of pooled |
| `--head_warmup_epochs` | linear-probe the fresh 200-class head before sparse adaptation |
| `--num_grad_batches`, `--adv_steps` | adversarial batches / PGD steps for the MRC estimate |
| `--pgd_steps`, `--pgd_restarts`, `--eval_batches` | evaluation PGD settings |
| `--diagnostic_only` | print inventory + caveats, then exit |
| `--allow_timm_fallback`, `--xcit_state_dict` | non-robustbench XCiT loading |

## Default checkpoints (per the research verification)

| arch / dataset | RobustBench `model_name` | eps | note |
|---|---|---|---|
| WRN-28-10 / CIFAR-10 | `Pang2022Robustness_WRN28_10` | 8/255 | lightest synthetic (+1M); see caveat |
| WRN-28-10 / CIFAR-100 | `Pang2022Robustness_WRN28_10` | 8/255 | +1M synthetic |
| XCiT-S12 / Tiny-ImageNet | `Debenedetti2022Light_XCiT-S12` (ImageNet) | 4/255 | head reset 1000→200 |

## Caveats baked into the run (printed at startup; footnote them in the paper)

1. **No pure-data WRN-28-10 exists in RobustBench** — every leaderboard WRN-28-10 uses
   DDPM/EDM synthetic data (`additional_data=False` does **not** mean "no synthetic").
   Default `Pang2022Robustness_WRN28_10` is the lightest (+1M); a `[DATA-CONDITION WARNING]`
   is printed. For a strict pure-AT match, switch `--model_name` to a pure-AT model at a
   different width (e.g. `Wu2020Adversarial`, `Gowal2020Uncovering_70_16`).
2. **Tiny-ImageNet via XCiT is transfer, not same-condition.** The 200-class head is fresh
   and non-robust; frame results as "transfer + sparse adaptation", use `--head_warmup_epochs`.
3. **Resolution shift**: Tiny-ImageNet 64→224 (bicubic) — absolute numbers aren't directly
   comparable to native ImageNet/CIFAR.
4. **Eps differs by dataset**: CIFAR 8/255, Tiny/ImageNet 4/255. Footnote eps/steps/restarts.

## Outputs (under `./results/{arch}_{dataset}/robustbench/`)

`output.log`, `neuron_masks.pth`, `neuron_mrc_list.npy`, `experiment_record.json`
(per-epoch clean/robust acc + final + selection metadata), `best_params.pth`.

**Please send back**: the `--diagnostic_only` inventory for each backbone, plus
`output.log` / `experiment_record.json` after a run, so I can verify the qkv/head
targeting and the numbers.
