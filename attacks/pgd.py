import torch
import torch.nn.functional as F


def pgd_targeted(model, x, target, k, eps, eps_step): # (target model, input image, target cls, num steps, eps, step size)
    was_training = model.training
    model.eval()

    x = x.detach().clone()
    x_adv = x.clone()
    target = target.to(x.device)

    for _ in range(k):
        x_adv.requires_grad_(True)

        model.zero_grad()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        # minimize loss for target cls
        x_adv = x_adv - eps_step * x_adv.grad.sign()

        # projection to eps boundary
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    if was_training:
        model.train()

    return x_adv


def pgd_untargeted(model, x, label, k, eps, eps_step): # (target model, input image, true cls, num steps, eps, step size)
    was_training = model.training
    model.eval()

    x = x.detach().clone()
    x_adv = x.clone()
    label = label.to(x.device)

    # iterative
    for _ in range(k):
        x_adv.requires_grad_(True)

        model.zero_grad()
        logits = model(x_adv)
        loss = F.cross_entropy(logits, label)
        loss.backward()

        # maximize loss for true cls
        x_adv = x_adv + eps_step * x_adv.grad.sign()

        # projection to eps boundary
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + delta, 0.0, 1.0).detach()

    if was_training:
        model.train()

    return x_adv
