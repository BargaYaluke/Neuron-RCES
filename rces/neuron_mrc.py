"""
Neuron-level Robust Criticality (NRC / MRC) for RobustBench backbones.

Two steps, faithful to the original main.py implementation:

  1. accumulate_adv_param_grads(): craft PGD adversarial samples in [0,1] space
     (the model self-normalizes internally -> NO extra normalization, NO wrapper)
     and accumulate the averaged robust-loss gradient into param.grad.

  2. compute_neuron_masks(): per weight tensor, per output unit, compute
        MRC = ||grad_row||_2 / (||weight_row||_2 + eps)
     and select the k SMALLEST-MRC units per layer as trainable
     (mask == 0 trainable, mask == 1 frozen).

Granularity:
  - Conv2d weight [C_out, C_in, kH, kW] : unit = output channel (dim0).
  - Linear weight [out, in]             : unit = output row   (dim0).
  - Fused qkv Linear [3*dim, in] with --vit_neuron_granularity head :
       group rows into attention heads. Verified timm-XCiT layout is
       rows = [3 (Q,K,V)] x [num_heads] x [head_dim] (3 outermost), so
       Q = rows[0:dim], K = rows[dim:2dim], V = rows[2dim:3dim], and head h
       within a block = rows[h*hd:(h+1)*hd], hd = dim // num_heads.

Everything else (bias, 1-D BatchNorm/LayerNorm affine, params without grad) is
fully frozen, exactly as the original method freezes them.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Step 1: accumulate adversarial robust-loss gradients into param.grad
# --------------------------------------------------------------------------- #
def accumulate_adv_param_grads(model, loader, device, eps, steps=10,
                               num_grad_batches=10, criterion=None,
                               alpha=None, random_start=True, logger=None):
    """
    Craft PGD adversarial examples (in [0,1]) and accumulate the averaged
    cross-entropy gradient over `num_grad_batches` batches into model params'
    .grad. Leaves param.grad populated for compute_neuron_masks().

    Returns the number of batches actually used.
    """
    log = (logger.info if logger is not None else print)
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    if alpha is None:
        alpha = 2.5 * eps / max(steps, 1)

    for p in model.parameters():
        p.requires_grad = True   # guarantee grads are populated for every weight
        p.grad = None
    model.eval()  # BN/LN in eval; grads still flow to weights

    used = 0
    for x, y in loader:
        if used >= num_grad_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ---- input-space PGD to build the adversarial batch (no param grads) ----
        x_adv = x.clone().detach()
        if random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
            x_adv = x_adv.clamp(0.0, 1.0)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            loss = criterion(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp_(0.0, 1.0)

        # ---- forward/backward on the adv batch to ACCUMULATE param grads ----
        out = model(x_adv.detach())
        loss = criterion(out, y) / float(num_grad_batches)
        loss.backward()
        used += 1

    log(f"[nrc] accumulated adversarial grads over {used} batch(es) "
        f"(eps={eps:.5f}, alpha={alpha:.5f}, steps={steps}).")

    # quick NaN/Inf guard (mirrors the original grad-check)
    bad = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            bad += 1
            log(f"[nrc][GRAD CHECK] {name}: nan/inf in grad "
                f"(max|g|={p.grad.abs().max().item():.3e})")
    if bad:
        log(f"[nrc][GRAD CHECK] {bad} param tensor(s) had nan/inf grads.")
    return used


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def infer_num_heads(model):
    """Read num_heads from a timm-XCiT-style model (model.blocks[0].attn)."""
    m = model.module if hasattr(model, "module") else model
    try:
        blocks = getattr(m, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            attn = getattr(blocks[0], "attn", None)
            if attn is not None and hasattr(attn, "num_heads"):
                return int(attn.num_heads)
    except Exception:
        pass
    return None


def _select_smallest_k(mrc, valid_mask, k):
    """Indices of the k smallest-MRC VALID units (invalid -> +inf, never picked)."""
    if int(valid_mask.sum()) == 0:
        return np.array([], dtype=int)
    mrc_sel = mrc.copy()
    mrc_sel[~valid_mask] = np.inf
    kk = int(min(k, int(valid_mask.sum())))
    return np.argsort(mrc_sel)[:kk]


def _head_mrc_and_mask(weight, grad, num_heads, k, eps, per_proj_heads=False):
    """
    Head-granularity selection for a fused qkv weight/grad of shape [3*dim, in].
    Returns (new_mask float32 [3*dim, in], selected_info list).
    mask: 0 = trainable, 1 = frozen.
    """
    p3, _ = weight.shape
    assert p3 % 3 == 0, f"qkv first dim {p3} not divisible by 3"
    dim = p3 // 3
    assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
    hd = dim // num_heads
    new_mask = np.ones_like(weight, dtype=np.float32)

    def rows(proj, head):
        s = proj * dim + head * hd
        return slice(s, s + hd)

    if not per_proj_heads:
        # one unit per head h: pool Q_h, K_h, V_h rows jointly.
        head_mrc = np.empty(num_heads, dtype=np.float64)
        for h in range(num_heads):
            g = np.concatenate([grad[rows(pp, h)] for pp in range(3)], axis=0)
            w = np.concatenate([weight[rows(pp, h)] for pp in range(3)], axis=0)
            gn, wn = np.linalg.norm(g), np.linalg.norm(w)
            head_mrc[h] = (gn / (wn + eps)) if (wn > 0 and gn > 0) else np.inf
        valid = np.isfinite(head_mrc)
        order = [h for h in np.argsort(head_mrc) if valid[h]]
        sel = order[:int(min(k, len(order)))]
        for h in sel:
            for pp in range(3):
                new_mask[rows(pp, h), :] = 0.0
        info = [(int(h), float(head_mrc[h])) for h in sel]
        return new_mask, info

    # separate Q/K/V heads: 3*num_heads candidate units.
    units = []
    for pp in range(3):
        for h in range(num_heads):
            g, w = grad[rows(pp, h)], weight[rows(pp, h)]
            gn, wn = np.linalg.norm(g), np.linalg.norm(w)
            m = (gn / (wn + eps)) if (wn > 0 and gn > 0) else np.inf
            if np.isfinite(m):
                units.append((m, pp, h))
    units.sort(key=lambda t: t[0])
    sel = units[:int(min(k, len(units)))]
    for m, pp, h in sel:
        new_mask[rows(pp, h), :] = 0.0
    info = [(int(pp), int(h), float(m)) for m, pp, h in sel]
    return new_mask, info


# --------------------------------------------------------------------------- #
# Step 2: compute the per-tensor neuron masks
# --------------------------------------------------------------------------- #
def compute_neuron_masks(model, neurons_per_layer=2, eps=1e-12,
                         vit_neuron_granularity="unit", num_heads=None,
                         head_target_substrings=("qkv",),
                         per_proj_heads=False,
                         always_train_substrings=(),
                         device="cuda", logger=None, verbose=False):
    """
    Build the gradient-gating masks from the accumulated adversarial grads.

    Returns:
      new_masks: {param_name: tensor (0 trainable / 1 frozen)}
      statistic: {param_name: [trainable_count, total_count]}
      neuron_mrc_list: [(param_name, unit_idx, mrc)]  (sorted ascending)
      selected: {param_name: [(unit_idx, mrc), ...] or head info}
    """
    log = (logger.info if logger is not None else print)
    new_masks, statistic, neuron_mrc_list, selected = {}, {}, [], {}

    if vit_neuron_granularity == "head" and num_heads is None:
        num_heads = infer_num_heads(model)
        if num_heads is None:
            log("[nrc][WARN] head granularity requested but num_heads could not "
                "be inferred (no model.blocks[0].attn.num_heads); pass --num_heads "
                "explicitly. Degrading to unit-level granularity for this run.")
            vit_neuron_granularity = "unit"

    for name, param in model.named_parameters():
        w = param.data.detach().cpu().numpy()

        # (0) forced fully-trainable override (e.g. a freshly-reset head).
        if any(s in name for s in always_train_substrings):
            m = np.zeros_like(w, dtype=np.float32)
            new_masks[name] = torch.from_numpy(m).to(device)
            statistic[name] = [int(m.size), int(m.size)]
            continue

        # (1) no grad, or not a ".weight", or unsupported ndim -> fully frozen.
        if param.grad is None or not name.endswith(".weight"):
            m = np.ones_like(w, dtype=np.float32)
            new_masks[name] = torch.from_numpy(m).to(device)
            statistic[name] = [0, int(m.size)]
            continue

        grad = param.grad.data.detach().cpu().numpy()

        is_fused_qkv = (grad.ndim == 2
                        and grad.shape[0] % 3 == 0
                        and any(name.endswith(s + ".weight") for s in head_target_substrings))

        if vit_neuron_granularity == "head" and is_fused_qkv and num_heads:
            new_mask, info = _head_mrc_and_mask(
                w, grad, num_heads, neurons_per_layer, eps, per_proj_heads)
            selected[name] = info
            for entry in info:
                # log per-head MRC under a synthetic index for traceability
                h = entry[0] if not per_proj_heads else entry[1]
                mrc = entry[-1]
                neuron_mrc_list.append((name + ":head", int(h), float(mrc)))

        elif grad.ndim in (2, 4):
            b = grad.shape[0]
            grad_2d = grad.reshape(b, -1)
            weight_2d = w.reshape(b, -1)
            grad_norm = np.linalg.norm(grad_2d, axis=1)
            weight_norm = np.linalg.norm(weight_2d, axis=1)
            mrc = grad_norm / (weight_norm + eps)
            for j in range(b):
                neuron_mrc_list.append((name, int(j), float(mrc[j])))
            valid = (weight_norm > 0) & (grad_norm > 0)
            sel = _select_smallest_k(mrc, valid, neurons_per_layer)
            new_mask = np.ones_like(w, dtype=np.float32)
            if grad.ndim == 4:
                new_mask[sel, :, :, :] = 0.0
            else:
                new_mask[sel, :] = 0.0
            selected[name] = [(int(i), float(mrc[i])) for i in sel]
        else:
            new_mask = np.ones_like(w, dtype=np.float32)

        trainable = int(new_mask.size - np.count_nonzero(new_mask))
        statistic[name] = [trainable, int(new_mask.size)]
        new_masks[name] = torch.from_numpy(new_mask).to(device)
        if verbose:
            pct = 100.0 * trainable / max(new_mask.size, 1)
            log(f"  {name}: {trainable}/{new_mask.size} ({pct:.4f}%) {new_mask.shape}")

    neuron_mrc_list = [t for t in neuron_mrc_list if t[2] > 0]
    neuron_mrc_list.sort(key=lambda x: x[2])

    total_trainable = sum(v[0] for v in statistic.values())
    total_params = sum(v[1] for v in statistic.values())
    log(f"[nrc] selected granularity='{vit_neuron_granularity}', "
        f"k={neurons_per_layer}/layer, num_heads={num_heads}; "
        f"trainable {total_trainable}/{total_params} "
        f"({100.0 * total_trainable / max(total_params, 1):.4f}%).")
    return new_masks, statistic, neuron_mrc_list, selected
