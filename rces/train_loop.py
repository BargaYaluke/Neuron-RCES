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
                    epoch, logger, log_interval=50):
    """One epoch of clean-data sparse adaptation with gradient gating."""
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    n = len(loader)
    start = time.time()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        if neuron_masks is not None:
            apply_grad_mask(model, neuron_masks)
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
        correct += (out.argmax(1) == y).sum().item()

        if (i % log_interval == 0) or (i == n - 1):
            logger.info(
                f"Adapt Epoch[{epoch}] [{i}/{n}] "
                f"loss {total_loss / max(total, 1):.4f} "
                f"acc {100.0 * correct / max(total, 1):.2f}% "
                f"({time.time() - start:.1f}s)")

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


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
        total_loss, total, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total += bs
            correct += (out.argmax(1) == y).sum().item()
        log(f"[warmup head] epoch {e} loss {total_loss / max(total, 1):.4f} "
            f"acc {100.0 * correct / max(total, 1):.2f}%")

    for p in model.parameters():
        p.requires_grad = True
