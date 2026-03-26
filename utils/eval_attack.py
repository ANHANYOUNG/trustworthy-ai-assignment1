import time as timer

import torch
from tqdm import tqdm


def eval_attack_targeted(
    model,
    data_loader,
    attack_fn,
    *,
    device,
    target_cls=0,
    num_samples=100,
    desc="targeted attack",
    **attack_kwargs,
):
    was_training = model.training
    model.eval()

    success = 0
    total = 0
    start_time = timer.perf_counter()

    eval_bar = tqdm(data_loader, desc=desc)
    for images, labels in eval_bar:
        remaining = num_samples - total
        if remaining <= 0:
            break

        # only consider target
        labels = labels[:remaining].to(device)
        target_mask = labels != target_cls
        if target_mask.sum().item() == 0:
            continue

        images = images[:remaining].to(device)[target_mask]
        labels = labels[target_mask]
        target = torch.full_like(labels, target_cls)

        x_adv = attack_fn(model, images, target, **attack_kwargs)

        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)

        # targeted success if pred == target
        success += (preds == target).sum().item()
        total += labels.size(0)

        eval_bar.set_postfix(success_rate=f"{success / total:.4f}")

    if was_training:
        model.train()

    success_rate = success / total
    time = timer.perf_counter() - start_time

    return success_rate, time


def eval_attack_untargeted(
    model,
    data_loader,
    attack_fn,
    *,
    device,
    num_samples=100,
    desc="untargeted attack",
    **attack_kwargs,
):
    was_training = model.training
    model.eval()

    success = 0
    total = 0
    start_time = timer.perf_counter()

    eval_bar = tqdm(data_loader, desc=desc)
    for images, labels in eval_bar:
        remaining = num_samples - total
        if remaining <= 0:
            break

        images = images[:remaining].to(device)
        labels = labels[:remaining].to(device)

        x_adv = attack_fn(model, images, labels, **attack_kwargs)

        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)

        # untargeted success if pred != label
        success += (preds != labels).sum().item()
        total += labels.size(0)

        eval_bar.set_postfix(success_rate=f"{success / total:.4f}")

    if was_training:
        model.train()

    success_rate = success / total
    time = timer.perf_counter() - start_time

    return success_rate, time
