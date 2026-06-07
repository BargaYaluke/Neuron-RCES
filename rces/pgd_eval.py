"""
Clean + PGD robustness evaluation for self-normalizing RobustBench backbones.

The model is attacked DIRECTLY (it normalizes inputs internally), and PGD
perturbs / projects / clamps in [0,1] pixel space — the eps-ball is defined on
raw pixels, before the model's internal normalization. No nn.Sequential(Normalize,
model) wrapper, no torchattacks dependency.

Per-dataset Linf budget (footnote these in any results table):
    CIFAR-10/100         eps = 8/255
    ImageNet / XCiT      eps = 4/255   (Tiny-ImageNet via XCiT keeps 4/255)
"""

import torch
import torch.nn.functional as F

EPS_BY_DATASET = {
    "cifar10": 8.0 / 255.0,
    "cifar100": 8.0 / 255.0,
    "imagenet": 4.0 / 255.0,
    "tinyimagenet": 4.0 / 255.0,
}


@torch.no_grad()
def clean_accuracy(model, loader, device, max_batches=None):
    model.eval()
    correct, total = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def pgd_accuracy(model, loader, device, eps, alpha=None, steps=50, restarts=1,
                 max_batches=None, logger=None):
    """
    Worst-case-over-restarts PGD accuracy. A sample counts as robust-correct only
    if it stays correctly classified under EVERY restart.
    """
    model.eval()
    if alpha is None:
        alpha = 2.5 * eps / max(steps, 1)
    correct, total = 0, 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        survives = torch.ones(y.size(0), dtype=torch.bool, device=device)

        for _ in range(max(restarts, 1)):
            x_adv = x.clone().detach()
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
            x_adv = x_adv.clamp(0.0, 1.0)
            for _ in range(steps):
                x_adv.requires_grad_(True)
                loss = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss, x_adv)[0]
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp_(0.0, 1.0)
            with torch.no_grad():
                pred = model(x_adv).argmax(1)
            survives &= (pred == y)

        correct += survives.sum().item()
        total += y.size(0)

    acc = 100.0 * correct / max(total, 1)
    if logger is not None:
        logger.info(f"[pgd] eps={eps:.5f} alpha={alpha:.5f} steps={steps} "
                    f"restarts={restarts} -> robust acc {acc:.2f}% "
                    f"({correct}/{total})")
    return acc


def autoattack_accuracy(model, dataset, n_examples, eps, device,
                        threat_model="Linf", data_dir="./data", logger=None):
    """
    Optional AutoAttack evaluation via robustbench.benchmark (if installed).
    Returns (clean_acc, robust_acc) in [0,1] as RobustBench reports, or None.
    """
    log = (logger.info if logger is not None else print)
    try:
        from robustbench import benchmark
    except ImportError:
        log("[autoattack] robustbench not installed; skipping AutoAttack.")
        return None
    rb_dataset = "imagenet" if dataset == "tinyimagenet" else dataset
    log(f"[autoattack] benchmark(dataset={rb_dataset}, n={n_examples}, "
        f"eps={eps:.5f}, threat_model={threat_model})")
    clean_acc, robust_acc = benchmark(
        model, n_examples=n_examples, dataset=rb_dataset,
        threat_model=threat_model, eps=eps, device=device,
        data_dir=data_dir, to_disk=False)
    return clean_acc, robust_acc
