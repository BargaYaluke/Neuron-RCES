"""
Startup module inventory.

Prints every nn.Linear / nn.Conv2d in the loaded backbone with its shape and a
guessed role (qkv / q / k / v / proj / mlp_fc1 / mlp_fc2 / patch_embed / head /
lpi_conv / conv / other). The output is copy-pasteable so it can be returned to
refine --head_target_substrings / --num_heads against the actual model.
"""

import torch.nn as nn


def _guess_role(name, mod):
    n = name.lower()
    if isinstance(mod, nn.Conv2d):
        if "local_mp" in n or "lpi" in n:
            return "lpi_conv"
        if "patch_embed" in n:
            return "patch_embed_conv"
        if "shortcut" in n or "downsample" in n or "convshortcut" in n:
            return "shortcut_conv"
        return "conv"
    # Linear roles
    if "attn.qkv" in n or n.endswith(".qkv"):
        return "qkv"
    if n.endswith("attn.q") or n.endswith(".q"):
        return "q"
    if n.endswith("attn.k") or n.endswith(".k"):
        return "k"
    if n.endswith("attn.v") or n.endswith(".v"):
        return "v"
    if "attn.proj" in n or n.endswith(".proj"):
        return "proj"
    if "mlp.fc1" in n or n.endswith(".fc1"):
        return "mlp_fc1"
    if "mlp.fc2" in n or n.endswith(".fc2"):
        return "mlp_fc2"
    if "patch_embed" in n:
        return "patch_embed"
    if n == "head" or n.endswith(".head") or "classifier" in n or n.endswith(".fc"):
        return "head"
    return "other"


def print_module_inventory(model, logger=None):
    """Print the Linear/Conv inventory and a summary. Returns the list of dicts."""
    log = (logger.info if logger is not None else print)
    m = model.module if hasattr(model, "module") else model

    rows = []
    n_linear = n_conv = n_fused_qkv = 0
    log("=" * 78)
    log("[inventory] Linear / Conv2d modules (name | type | weight.shape | role)")
    log("-" * 78)
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            n_linear += 1
            shape = tuple(mod.weight.shape)  # [out, in]
            role = _guess_role(name, mod)
            if role == "qkv" and shape[0] % 3 == 0:
                n_fused_qkv += 1
            rows.append({"name": name, "type": "Linear", "shape": shape,
                         "out": shape[0], "in": shape[1], "role": role})
            log(f"  {name:<46} Linear  {str(shape):<16} {role}")
        elif isinstance(mod, nn.Conv2d):
            n_conv += 1
            shape = tuple(mod.weight.shape)  # [out, in, kH, kW]
            role = _guess_role(name, mod)
            rows.append({"name": name, "type": "Conv2d", "shape": shape,
                         "out": shape[0], "in": shape[1], "role": role})
            log(f"  {name:<46} Conv2d  {str(shape):<16} {role}")

    # num_heads / head_dim detection for ViT-style models
    num_heads = head_dim = None
    try:
        blocks = getattr(m, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            attn = getattr(blocks[0], "attn", None)
            if attn is not None and hasattr(attn, "num_heads"):
                num_heads = int(attn.num_heads)
    except Exception:
        pass
    if num_heads:
        for r in rows:
            if r["role"] == "qkv":
                head_dim = (r["out"] // 3) // num_heads
                break

    log("-" * 78)
    log(f"[inventory] totals: Linear={n_linear}, Conv2d={n_conv}, "
        f"fused_qkv={n_fused_qkv}, num_heads={num_heads}, head_dim={head_dim}")
    log("=" * 78)
    return rows
