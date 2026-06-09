"""
Gradient-masked sparse adaptation loop (Neuron-RCES) + optional linear-probe
warm-up for the transfer (Tiny-ImageNet) case.

Mask convention (identical to the original main.py):
    mask == 0 -> trainable, mask == 1 -> frozen
    gating:  param.grad.mul_(1.0 - mask)   (applied AFTER backward, BEFORE step)
"""

import time
import torch


def apply_grad_mask(model, neuron_masks):
    """Zero the gradient of frozen entries (mask==1). Caches device-moved masks."""
    for name, p in model.named_parameters():
        if p.grad is None or name not in neuron_masks:
            continue
        mask = neuron_masks[name]
        if mask.device != p.grad.device:
            mask = mask.to(p.grad.device)
            neuron_masks[name] = mask
        assert mask.shape == p.grad.shape, (
            f"{name}: mask {tuple(mask.shape)} != grad {tuple(p.grad.shape)}")
        p.grad.mul_(1.0 - mask)


def adapt_one_epoch(model, loader, optimizer, criterion, neuron_masks, device,
                    epoch, logger, log_interval=50, scaler=None):
    """One epoch of clean-data sparse adaptation with gradient gating."""
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()

    # GPU-resident accumulators: summing on the GPU avoids the per-batch .item()
    # CUDA sync that otherwise stalls the input pipeline. Synced once per log point.
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0

    # Precompute (param, keep = 1 - mask) ONCE so the hot loop has no
    # named_parameters() walk, dict lookup, 1-mask rebuild or shape assert.
    mask_pairs = []
    if neuron_masks is not None:
        for name, p in model.named_parameters():
            m = neuron_masks.get(name)
            if m is None:
                continue
            assert tuple(m.shape) == tuple(p.shape), (
                f"{name}: mask {tuple(m.shape)} != param {tuple(p.shape)}")
            mask_pairs.append((p, (1.0 - m.to(device)).to(dtype=p.dtype)))

    n = len(loader)
    start = time.time()

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # A 0/1 mask commutes with GradScaler's positive scalar, so masking the
        # (possibly scaled) grads here is exact (mask==0 trainable, mask==1 frozen).
        if mask_pairs:
            with torch.no_grad():
                for p, keep in mask_pairs:
                    if p.grad is not None:
                        p.grad.mul_(keep)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        bs = y.size(0)
        total_loss += loss.detach().double() * bs
        total += bs
        correct += (out.argmax(1) == y).sum()

        if (i % log_interval == 0) or (i == n - 1):
            loss_val = (total_loss / max(total, 1)).item()
            acc_val = (100.0 * correct.double() / max(total, 1)).item()
            logger.info(
                f"Adapt Epoch[{epoch}] [{i}/{n}] "
                f"loss {loss_val:.4f} "
                f"acc {acc_val:.2f}% "
                f"({time.time() - start:.1f}s)")

    return (total_loss / max(total, 1)).item(), (100.0 * correct.double() / max(total, 1)).item()


def linear_probe_epochs(model, loader, criterion, device, epochs, lr,
                        head_substrings=("head",), weight_decay=1e-4,
                        logger=None):
    """
    Warm up a freshly-reset classifier head (backbone frozen) before sparse
    adaptation. Needed for the transfer case where the new head is random/
    non-robust. Re-enables requires_grad on all params at the end so the
    subsequent gradient-mask step controls trainability.
    """
    log = (logger.info if logger is not None else print)

    for name, p in model.named_parameters():
        p.requires_grad = any(s in name for s in head_substrings)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        log("[warmup] no head params matched; skipping linear-probe warm-up.")
        for p in model.parameters():
            p.requires_grad = True
        return

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    model.train()
    for e in range(epochs):
        # GPU-resident accumulators: this loop previously synced TWICE every batch
        # with no log gating — the worst offender. Sync once per epoch now.
        total_loss = torch.zeros((), device=device, dtype=torch.float64)
        correct = torch.zeros((), device=device, dtype=torch.long)
        total = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            bs = y.size(0)
            total_loss += loss.detach().double() * bs
            total += bs
            correct += (out.argmax(1) == y).sum()
        log(f"[warmup head] epoch {e} loss {(total_loss / max(total, 1)).item():.4f} "
            f"acc {(100.0 * correct.double() / max(total, 1)).item():.2f}%")

    for p in model.parameters():
        p.requires_grad = True
