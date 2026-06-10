# Directional Ablation — does the NRC *direction* matter, or just the budget?

**Setting:** ResNet-18 / CIFAR-10 only. One variable: the **rule that picks which k
neurons per layer are unfrozen**. Everything else — the attribution, the budget, the
fine-tuning — is held fixed.

The claim being tested: Neuron-RCES works because it unfreezes the **least robust-critical**
neurons (lowest NRC). A reviewer's first objection is "maybe *any* k-per-layer unfreezing
fine-tunes just as well." This table answers it by flipping only the selection direction.

```
NRC_{l,j} = ‖G_{l,j}‖₂ / (‖W_{l,j}‖₂ + ε)      computed ONCE, cached, shared by all arms
arm = which k_l neurons of layer l we unfreeze, given that ONE ranking  ← the only variable
```

| arm | rule (per layer, over the SAME alive set) | role |
|-----|-------------------------------------------|------|
| **lowest** | k smallest NRC | **the main method** (reference) |
| highest | k largest NRC | train the units we *claim* are most critical |
| random | k uniform w/o replacement (independent `mask_seed`) | chance baseline |

## The core design rule: one attribution, three arms

The attribution is run **exactly once**, on the AT checkpoint (RobustBench RN18/CIFAR-10,
loaded by `create_model` + `load_sd`), with the **same fixed adversarial subset B, same PGD
params, same input-normalization convention** as the main table. That single
`main.py --cal_neuron_mrc` run caches:

- `neuron_mrc_list.npy` — the per-neuron NRC scores for every **alive** neuron
  (`‖W‖₂>0 ∧ ‖grad‖₂>0`). This **is** the per-layer score tensor on disk, and its key set
  **is** the shared candidate set `S_l`.
- `neuron_masks.pth` — the canonical *lowest* mask, reused downstream **only** as a per-layer
  shape + budget template (`k_l` = its trainable channel count).

All three arms derive their masks from that one cache (`derive_mask.py`); **nothing is ever
re-attributed.** This is the whole point:

> If each arm recomputed attribution, the run-to-run wobble of PGD / gradient accumulation
> would ride on top of the selection rule and contaminate the only axis we are varying.

(Contrast the normalizer ablation, which *does* re-run selection per cell — it can afford to
because it argues the numerator is byte-identical across modes. The directional ablation does
not take that risk; it freezes the numbers to disk.)

## Equal budget, down to the neuron

`k_l` is read back from the main method's **own** output (the template's per-layer trainable
count), and every arm picks exactly `k_l` channels from the **same** alive set `S_l`. So:

- total trainable parameters are identical across arms, **layer by layer, neuron by neuron**
  → the "different parameter count" confound is removed by construction;
- `dir_metrics.py` asserts this (`[equal-count OK]`) and voids the comparison if it ever fails.

## The three boundary conventions (pinned, identical across arms)

You asked these be written down and held constant. They are **inherited automatically**,
because both the candidate set `S_l` and the budget `k_l` come from the main method's output:

1. **Classification head** — `linear.weight` is 2-D, so it is an eligible layer: its
   class-rows are candidate neurons and participate in selection, the same way for all arms.
   `linear.bias` is 1-D → frozen. (This is exactly what `neuron_mrc_and_prune` already does.)
2. **BatchNorm affine** — `*.bn*.weight` / `*.bias` are 1-D → the frozen branch → frozen for
   all arms.
3. **N_l < k** — `k_l = min(k, |S_l|)` as the main method computed it; all arms shrink
   identically. (For ResNet-18 / k=4 every layer has `|S_l| ≫ 4`, incl. the 10-way head, so
   this never actually triggers — but it is handled.)

**Candidate-set note.** `random` samples from `S_l` (the alive set the other two arms rank
over), *not* from all `N_l` channels, so the candidate set is byte-identical across arms — the
control you specified ("唯一变化的是被选中的那 k 个神经元"). For RN18/CIFAR-10 essentially
every channel is alive, so `S_l == ` all `N_l` and this is a no-op; pass
`--random_pool all` to sample from every channel instead.

## Training protocol — cloned from the main method

Clean CE objective, **γ=1, no consolidation** (compare the un-interpolated fine-tuned model,
i.e. `final_result`, not the RiFT α-sweep), and `epochs / lr / batch_size / optimizer /
weight-decay handling / seed` all identical to the main-table Neuron-RCES cell. The **same
training seed** is shared by all arms (fine-tuning randomness is a common term, not the
variable); the **random arm gets several independent `mask_seed`s** to show its distribution,
while lowest / highest are deterministic.

## How to run

```bash
bash run_dir_ablation.sh          # edit CKPT + hyperparams at the top to match the main table
```

It runs selection **once** → `derive_mask.py` for `{lowest, highest, random×mask_seeds}` →
fine-tune each arm (gradient-gating, identical hyperparams) → `dir_metrics.py` prints the
joint table.

The table alone (no re-train) can be regenerated any time:

```bash
python dir_metrics.py \
  --cell lowest=<dir> --cell highest=<dir> --cell random=<dir> ... \
  --ref lowest --out dir_ablation_table.json
```

## Results table template

| arm | #sel (must match) | Jaccard vs lowest (micro/macro) | Clean Acc % | Robust Acc % |
|---|---|---|---|---|
| **lowest (ours)** | _ | 1.000 / 1.000 | _ | _ |
| highest | _ (same) | ~0 / ~0 | _ | _ |
| random (mean±std over mask seeds) | _ (same) | ~k/N_l | _ | _ |

**How to read it.** The story you want is **lowest ≥ random > highest** on robust acc with
**identical budget**: training the *least*-critical units (lowest NRC) is what carries the
method, and deliberately training the *most*-critical units (highest) is actively worse —
so the NRC **direction** is the mechanism, not the mere act of unfreezing 4 neurons/layer.
The `lowest-vs-highest` Jaccard ≈ 0 confirms the two arms are disjoint ends of the **same**
cached ranking; `lowest-vs-random` ≈ chance confirms random shares neurons only by accident.
If instead all three tie on robust acc at equal budget, the honest conclusion is "on
RN18/CIFAR-10 the selection direction is second-order" — which still answers the reviewer.

## Artifacts per arm

`results/ResNet18_CIFAR10/checkpoint/rift_..._neurons=4_dir_<arm>_s<seed>/`

- `neuron_masks.pth` — the arm's selected sets + stamped `arm` / `mask_seed` / `derived_from`
  (proves all arms trace to one cached attribution)
- `experiment_record.json` → `final_result.{final_test_acc, final_robust_acc}` (γ=1 numbers)

The single shared cache lives in `..._dir_sel_s<seed>/`:
`neuron_mrc_list.npy` (scores) + `neuron_masks.pth` (lowest template).
