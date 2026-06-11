# γ-consolidation Ablation — is the gain NRC selection, or borrowed RiFT consolidation?

**Setting:** ResNet-18 / CIFAR-10. One variable: the consolidation strength **γ** of the
RiFT-style interpolation in §3.4. Everything else (the NRC selection, the fine-tune, the
attack) is the finished main-table run — this sweep does **no training**.

```
Eq.(8):  θ_final = θ_initial + γ·(θ_tuned − θ_initial) ⊙ (1 − M̃)
                   └ AT start ┘         └ main-table model ┘   └ M̃=1 ⟺ frozen ┘
```

## What this is for (and what it is *not*)

Unlike the normalizer / directional / layer-wise ablations, this one tests **no property
of NRC**. It pays off a writing promise: §3.4 borrows RiFT's interpolation step, so a
reviewer will ask *"is your improvement actually RiFT's consolidation rather than the NRC
selection?"* This curve leaves that question nowhere to land. The argument is **defensive**.

There is exactly **one** thing to read off the figure:

> **γ=1 (no consolidation) must equal the main-table Neuron-RCES number, and no γ<1 point
> may beat it by more than seed noise.** If both hold, consolidation is an *optional
> post-processing knob*, not the source of the gain — the gain is the selection.

If some γ<1 *does* beat γ=1, that is not a disaster but it **must be reported**: it would
mean the main table is not the best consolidation point, and hiding it is exactly the
"private goods" a reviewer is probing for.

## The interpolation must match Eq. (8) item by item

`θ_initial` = `init_params.pth` (full `state_dict` saved *before* fine-tuning = the
AT/RobustBench start). `θ_tuned` = `best_params.pth["model"]` (the main-table model).
`M̃` = the stored `neuron_masks.pth["masks"]`, whose convention (`main.py`: *mask=0
trainable, mask=1 frozen*) **is already M̃** (M̃=1 ⟺ frozen). Three details, all enforced
or measured by `gamma_consolidation.py`:

**(a) Mask provenance.** The sweep reads the **landed** `neuron_masks.pth` that sat next to
the checkpoint during fine-tuning — it never re-derives M̃ from cached NRC scores. A later
edit to the scoring code therefore cannot drift the mask out from under a finished run.

**(b) The mask factor is redundant — and that is a free consistency check.** Frozen neurons
are pinned at θ_initial during training (gradient gating + excluded from weight decay), so
`θ_tuned − θ_initial` is already **zero** on M̃=1 positions and the `⊙(1−M̃)` factor is a
numerical no-op. So the script interpolates the **parameters un-masked** (a full lerp) —
which makes `γ=0 == θ_initial` and `γ=1 == θ_tuned` *bit-exact* — and **separately asserts**

```
‖(θ_tuned − θ_initial) ⊙ M̃‖ ≈ 0      (per-cell, printed as leak_max)
```

Applying the mask would *silently erase* any leak; refusing to apply it and asserting
instead is what makes a leak **visible**. A non-zero norm means a frozen weight moved — and
the scan **aborts** (`--force` to override). This must be understood *before* the curve
means anything.

> ⚠️ **This leak is the default, not an edge case.** `create_optimizer` (optimizer.py)
> puts *all* parameters in one group with a **global** weight decay. Under the default
> `mask_mode=grad`, the mask only zeroes frozen *gradients* — SGD still applies `wd·p` to
> every weight, so **frozen weights decay toward 0** and drift off θ_initial. The leak
> check *will* fire. The premise "frozen neurons stay at θ_initial" holds bit-exactly only
> under **`--mask_mode param`**, where `apply_param_freeze` (main.py) overwrites frozen
> positions back to `w0` after each optimizer step. **Recommendation: train the main-table
> run with `--mask_mode param`** (or give frozen params a `wd=0` group). If instead the
> main table used `mask_mode=grad`, γ=1 still equals it (the un-masked lerp keeps the
> drifted frozen weights), but Eq. (8)'s interior claim that frozen units are *held* at
> θ_initial is then only approximate — say so, or re-run with `param`.

**(c) BN running stats.** `running_mean`/`running_var` are **not** trainable parameters but
**do** update on every forward pass in `train()` mode — the gradient mask gates *gradients*,
not the BN EMA — so `θ_tuned`'s buffers ≠ `θ_initial`'s buffers in general. Interpolating
buffers has no theoretical basis, and using `θ_initial`'s buffers would make γ=1 miss the
main table. **Default policy: buffers follow `θ_tuned` at every γ** (`--bn_buffers tuned`).
Consequence: **γ=1 reproduces the main table exactly**; **γ=0 = init-weights + tuned-buffers**,
which equals the AT baseline *only up to BN drift*. The script therefore also evaluates the
full `θ_initial` ("AT anchor") and prints the **γ=0-vs-anchor gap**, so the drift is
quantified, not assumed away.

> ⚠️ **One internal tension, made explicit.** "Buffers = θ_tuned everywhere" (c) and
> "γ=0 *strictly* recovers the AT baseline" can both be bit-exact **only if** BN stats did
> not drift. They generally do. The default keeps the load-bearing endpoint exact (γ=1 ==
> main table) and **reports** the γ=0 deviation. If that deviation is noise-level, both
> checks pass as written. If it is not, choose one and say so in the paper:
> `--bn_buffers interp` (both endpoints bit-exact, interior buffers blended), or report the
> AT-anchor row as the γ=0 baseline and note the BN-drift caveat.
>
> Note: the repo's own `utils.interpolation()` does the *opposite* of (c) — it lerps the
> **entire** state dict, BN buffers included, and applies **no** mask (it is faithful only
> because the mask is redundant, but it blends buffers — Policy "interp"). The figure in the
> paper should come from `gamma_consolidation.py`, not from that helper.

## γ grid & evaluation protocol

- **Grid:** γ ∈ {0.0, 0.1, …, 0.9, 1.0} — 11 points. Each is **one pure-inference build &
  evaluate**, no training.
- **Two free endpoint checks** (printed every run):
  - **γ=0 vs AT anchor** — under `--bn_buffers tuned` this gap *is* the BN-drift effect (≈0
    if BN didn't move).
  - **γ=1 vs `experiment_record.json → final_result`** — confirms the runner is faithful to
    the main-table protocol (should match to rounding).
- **Metrics (three):**
  - **clean** — clean test accuracy (normalized test set).
  - **adv** — PGD-10, `eps=8/255` — the **same** attack as the main table
    (`evaluate_cifar_robustness`).
  - **ood** — CIFAR-10-C mean over 18 corruptions (`evaluate_cifar_corruption`).
- **Why keep OOD here** (the first two ablations could drop it): consolidation's entire
  reason for existing is the **stability–plasticity trade-off**. Clean / adv / ood on one
  axis show the *shape* of that trade-off; only together do they justify calling γ a knob.
- **Seeds:** pass one `--cell` per main-table seed checkpoint (3 by default). Every γ is
  evaluated in every cell; the table reports **mean±std** across cells.

## How to run

```bash
bash run_gamma_ablation.sh        # edit the header to match the main-table hyperparams/cells
```

It locates the 3 seed cells from the main-table fine-tune (each already holds
`init_params.pth`, `best_params.pth`, `neuron_masks.pth`, `experiment_record.json`) and runs
the scan. No re-training. Direct invocation / quick smoke test:

```bash
py gamma_consolidation.py \
  --cell <cell_s0> --cell <cell_s1> --cell <cell_s2> \
  --model ResNet18 --dataset CIFAR10 --num_classes 10 --input_size 32 \
  --out results/ResNet18_CIFAR10/checkpoint/gamma_consolidation.json
# fast: add  --gammas 0,0.5,1 --no-ood
# both endpoints bit-exact instead of the default: add  --bn_buffers interp
```

## Results table template

| γ | clean % | adv % (PGD-10) | OOD % (CIFAR-10-C) | note |
|--:|:-------:|:--------------:|:------------------:|------|
| 0.0 | _ | _ | _ | AT baseline (== anchor up to BN drift) |
| 0.1 | _ | _ | _ | |
| … | _ | _ | _ | |
| 0.9 | _ | _ | _ | |
| **1.0** | _ | _ | _ | **main table — no consolidation** |

**How to read it.** The story you want: the **γ=1** row equals your main table, and the
curve is flat-to-gently-sloping with **no γ<1 spike** above γ=1 on adv beyond ±std → the
consolidation step is a knob you *could* turn, not the engine. The script prints this verdict
(`[read] best gamma<1 on adv = … ; gamma=1 = …`). If clean rises and adv falls as γ→0 (a
clean stability-plasticity sweep) that is the *expected* RiFT shape and reinforces the point:
γ trades the two off, and you reported the operating point honestly.

## Artifacts per cell

`results/ResNet18_CIFAR10/checkpoint/<suffix>[_s<seed>]/`

- `init_params.pth` — θ_initial (AT start; full state_dict incl. BN buffers)
- `best_params.pth` — θ_tuned under `["model"]` (the main-table model)
- `neuron_masks.pth` — `["masks"]` is M̃ (1=frozen, 0=trainable); used **only** for the
  leak check (b), never to mask the lerp
- `experiment_record.json` — `final_result.{final_test_acc, final_robust_acc}` for the free
  γ=1 endpoint check

Output: `gamma_consolidation_<bn_buffers>.json` (full per-cell points + mean±std aggregate)
and `…txt` (the printed table).
