import torch
import torch.nn.functional as F


def fgsm_targeted(model, x, target, eps): # (target model, input image, target cls, eps)
    was_training = model.training         # check if train
    model.eval()                          # set to eval

    x = x.detach().clone()
    x.requires_grad_(True)
    target = target.to(x.device)

    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    # minimize loss for target cls
    x_adv = x - eps * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    if was_training:
        model.train()                      # back to train

    return x_adv


def fgsm_untargeted(model, x, label, eps): # (target model, input image, true cls, eps)
    was_training = model.training
    model.eval()

    x = x.detach().clone()
    x.requires_grad_(True)
    label = label.to(x.device)

    model.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, label)
    loss.backward()

    # maximize loss for true cls
    x_adv = x + eps * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    if was_training:
        model.train()

    return x_adv
