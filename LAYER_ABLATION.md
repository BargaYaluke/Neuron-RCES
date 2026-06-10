# Layer-scope Ablation — must selection be done *per layer*, or can NRC be pooled globally?

**Setting:** ResNet-18 / CIFAR-10 only. One variable: the **scope over which neurons are
pooled before the k-budget is spent**, plus the **scale on which they are compared**.
Everything else — the attribution, the total budget, the fine-tuning — is held fixed.

This is a *different* claim from the directional ablation. The directional ablation asks
"does the NRC **ranking direction** carry information?" (lowest vs highest vs random, each
selected **per layer**). This ablation asks the §3.3 design question: **NRC is comparable
*within* a layer but **not** *across* layers, so selection must proceed layer by layer.**

```
NRC_{l,j} = ‖G_{l,j}‖₂ / (‖W_{l,j}‖₂ + ε)      computed ONCE, cached, shared by all arms
arm = the POOL we rank over + the SCALE we rank on   ← the only variable
```

## The two-part argument (why one table is not enough)

This experiment is only convincing as a **pair** of results:

1. **Global-raw must fail.** If we drop the per-layer structure and pool every neuron's
   *raw* NRC into one global ranking, the lowest-budget picks should be badly skewed — the
   per-layer top-k coverage collapses and a few layers swallow the whole budget. This proves
   **"cross-layer incomparable" is a real problem**, not a strawman.
2. **Global-normalized must (mostly) recover.** If we first standardize NRC *within each
   layer* to a comparable scale, then pool globally, the result should be close to the main
   method. This proves the failure in (1) is caused by the **scale mismatch** between layers,
   **not** by the act of global selection itself.

Together they say: the mechanism (pick least-critical units) is sound globally too — the
**only** thing that breaks it is the un-normalized cross-layer scale. And since the
normalized-global arm recovers but buys nothing over the main method, **fixed-k-per-layer
still wins on simplicity and on guaranteed uniform coverage** — which is the §3.3 conclusion.

## The arms (equal *total* budget; per-layer occupancy is deliberately free)

Let there be `L` eligible layers. The main method takes the `k` lowest-NRC neurons **per
layer** → `B = Σ_l k_l` trainable neurons total (the running text uses `k=2 → B=2L`; the
scripts read the real `B` from `--budget_k`, so this generalizes to any `k`). All arms are
anchored to **exactly `B`**.

| arm | pool | scale | how `B` is spent |
|-----|------|-------|------------------|
| **layerwise** (ours) | per layer | raw NRC | `k` per layer, every layer → flat occupancy `c_l = k` |
| global-raw | whole net | raw NRC | globally lowest `B` — **no cross-layer correction** |
| global-z | whole net | within-layer **z-score** `(x−μ_l)/σ_l` | globally lowest `B` |
| global-rank | whole net | within-layer **rank/quantile** `(r+½)/N_l` | globally lowest `B` |

- **z-score** keeps each layer's distribution *shape* (only shifts/scales it), so it is the
  **more informative** normalized control — it still lets a layer with a genuinely heavy low
  tail contribute more.
- **rank** is the **most aggressive** scale removal (every layer is forced onto the same
  uniform `[0,1]` grid), so it should be the **flattest** arm. Running both is the robustness
  check: **if z-score and rank agree, the "it's just scale" conclusion is solid.**

> **One attribution, four arms.** The NRC scores are cached **once** (`--cal_neuron_mrc
> --norm_mode l2`, the same single run the directional ablation uses) and **every arm derives
> its mask from that one cache via `derive_layer_mask.py` — nothing is ever re-attributed.**
> The per-layer z-score / rank transforms are applied **on top of** the cached l2-NRC values
> at selection time; they do not touch the numerator. If the directional ablation has already
> run, this experiment costs **zero** extra attribution. (See the cache-reuse note below.)

## Equal *total* budget — and why this control differs from the directional one

`B` is identical across all four arms (`layer_metrics.py` asserts it and voids the table
otherwise). But — unlike the directional ablation, which pinned budget **per layer** — here
the **per-layer counts `c_l` are exactly what we let vary**. The control is therefore at the
**total** level only; the per-layer divergence is the *measurement*, reported as occupancy.

The candidate set is byte-identical across arms: every arm ranks/pools over the **same alive
set** `S_l = {j : ‖W_{l,j}‖₂>0 ∧ ‖G_{l,j}‖₂>0}` (exactly the `>0` filter `main.py` applies
before caching `neuron_mrc_list.npy`). `k` is decoupled from the cache: `derive_layer_mask.py`
recomputes `k_l = min(k, |S_l|)` and `B = Σ_l k_l` from the scores + `--budget_k`, using the
template **only for tensor shapes** (which are k-independent). So the directional cache (run
at any `k`) is reusable here verbatim.

## Boundary conventions (pinned, identical across arms)

These are written down and held constant — **including for the extreme cases global selection
can produce.** No special-casing; the consequences are exactly what the experiment exposes.

1. **Classification head** — `linear.weight` is 2-D → an eligible layer; its 10 class-rows are
   candidate neurons and compete in the **same global pool** as every conv neuron. `linear.bias`
   is 1-D → frozen. (Identical to `neuron_mrc_and_prune`.)
2. **BatchNorm affine** — `*.bn*.weight` / `*.bias` are 1-D → frozen branch → frozen for all arms.
3. **Extreme occupancy under global selection** — a global arm may select **0** neurons in a
   layer, or **all** `N_l` of a layer. `c_l > N_l` is mathematically impossible (each neuron is
   picked at most once). When a whole layer becomes trainable, its accompanying BN/bias params
   follow the **same** rule as for any trainable neuron under the main method — i.e. they stay
   frozen (they are 1-D). **No special case is introduced for the extreme regime** — we let the
   global arm's consequences show, because that is the thing the table is meant to reveal.

## Occupancy is dumped at mask-generation time (not after training)

The moment each arm's mask is built, `derive_layer_mask.py` writes `<dir>/occupancy.json`
holding, per layer in **depth order**:

```
{depth_idx, name, N_l, n_alive, k_template, c_l}
```

plus two scalars that go straight into the prose (more forceful than asking the reader to
eyeball the figure):

- **coverage** = `#{l : c_l > 0} / L` — fraction of layers that received *any* budget.
- **max single-layer share** = `max_l c_l / B` — how much of the whole budget the most
  occupied layer swallowed.

**Expected picture** (read straight off the per-layer `c_l` in each arm's `occupancy.json`;
sort layers by depth for the eventual figure — x = layer index by depth, y = `c_l`):

- **layerwise** — a flat line at height `k` (the reference).
- **global-raw** — severely skewed. By NRC's construction (‖grad‖ / ‖W‖) the mass should pile
  up at particular depths; **which depths is itself a result** and should cross-check against
  the NRC-distribution figure (Fig. 4). Expect low coverage, high max-share.
- **global-z** — markedly flattened vs raw, but still rippled (z-score preserves shape).
- **global-rank** — the flattest (rank erases scale entirely).

This occupancy data is the direct evidence for §3.3 reasons two (global selection concentrates
across layers) and three (extreme-distribution layers dominate); the accuracy table carries
reason one. (Plotting is intentionally left out of the pipeline — the raw `c_l` lands in
`occupancy.json` at mask-generation time, so the figure can be drawn separately whenever.)

## Training protocol — cloned from the directional ablation

Clean CE objective, **γ=1, no consolidation** (compare the un-interpolated `final_result`, not
the RiFT α-sweep), and `epochs / lr / wd / batch_size / optimizer / seed` all identical to the
main-table Neuron-RCES cell and to the directional ablation. The fine-tuning seed is shared by
all arms (training randomness is a common term, not the variable). Selection here is
**deterministic** for every arm — there is no `mask_seed` (no random arm).

## How to run

```bash
bash run_layer_ablation.sh      # edit CKPT + hyperparams at the top to match the main table
```

It (1) reuses the directional attribution cache if present, else runs selection **once**;
(2) `derive_layer_mask.py` for `{layerwise, global_raw, global_z, global_rank}`, each dumping
`occupancy.json`; (3) fine-tunes each arm (gradient-gating, identical hyperparams); (4)
`layer_metrics.py` prints the joint table (equal-budget control + coverage/max-share + accs).

The table (no re-train) can be regenerated any time:

```bash
python layer_metrics.py --cell layerwise=<dir> --cell global_raw=<dir> \
  --cell global_z=<dir> --cell global_rank=<dir> --ref layerwise --out layer_ablation_table.json
```

## Cache reuse (zero extra attribution)

`run_layer_ablation.sh` points `--scores` / `--template` at the directional ablation's
`..._dir_sel_s<seed>/` by default. Because the NRC scores and the tensor shapes are both
**k-independent**, that cache is valid here **whatever `k` the directional run used** — only
`--budget_k` (set at the top of this script) sets the budget. If the directional cache is
absent, the script runs the single `--cal_neuron_mrc --norm_mode l2` selection itself into
`..._layer_sel_s<seed>/`.

> **k note.** The running text uses `k=2`. The directional driver shipped with `NEURONS=4`.
> Pick whichever your **main table** uses and set `BUDGET_K` at the top of
> `run_layer_ablation.sh`; the scripts derive `B` from it and the cache is shared either way.
> The `layerwise` arm is re-derived + fine-tuned at `BUDGET_K` for self-containedness; set
> `LAYERWISE_DIR` to an existing main-method/lowest finetune at the **same** `k` to skip it.

## Results table template

| arm | pool / scale | #sel (must == B) | coverage `L_cov/L` | max share `max c_l/B` | Clean Acc % | Robust Acc % |
|---|---|---|---|---|---|---|
| **layerwise (ours)** | per layer / raw | _ | 1.000 | `k/B` (=1/L) | _ | _ |
| global-raw | global / raw | _ (same) | **low** | **high** | _ (expect ↓) | _ (expect ↓↓) |
| global-z | global / z-score | _ (same) | ~high | moderate | _ | _ (≈ ours) |
| global-rank | global / rank | _ (same) | highest | lowest | _ | _ (≈ ours) |

**How to read it.** The story is **global-raw ≪ {global-z ≈ global-rank ≈ layerwise}** on
robust acc at **identical total budget**: un-normalized global selection collapses (proving
cross-layer NRC is incomparable), normalized global selection recovers (proving the culprit is
scale, not global selection), and `layerwise` matches the recovered arms while being the
simplest rule with guaranteed uniform coverage. If `global-z` and `global-rank` agree, the
"it's just the scale" conclusion is robust to the choice of normalizer. If instead global-raw
*also* matches on RN18/CIFAR-10, the honest conclusion is "on this pair the per-layer scale gap
is second-order" — still a direct answer to the §3.3 design question.

## Artifacts per arm

`results/ResNet18_CIFAR10/checkpoint/rift_..._neurons=K_layer_<arm>_s<seed>/`

- `neuron_masks.pth` — the arm's selected sets + stamped `arm` / `pool_scope` / `norm` /
  `derived_from` (proves all arms trace to one cached attribution)
- `occupancy.json` — per-layer `c_l` (depth-ordered) + `coverage` + `max_share`, dumped at
  mask-generation time
- `experiment_record.json` → `final_result.{final_test_acc, final_robust_acc}` (γ=1 numbers)

The single shared cache lives in `..._dir_sel_s<seed>/` (directional) or
`..._layer_sel_s<seed>/`: `neuron_mrc_list.npy` (scores) + `neuron_masks.pth` (shape template).
