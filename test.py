from pathlib import Path

import torch

from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted
from models.cifar_resnet import CIFARResNet
from models.mnist_cnn import MNISTCNN
from utils.data import get_dataloaders
from utils.eval_attack import eval_attack_targeted, eval_attack_untargeted
from utils.train import train_classifier
from utils.vis import save_attack_visualization


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# save successful attack examples for visualization
def save_attack_result(
    model,
    dataset,
    *,
    device,
    attack_name,    # fgsm or pgd
    attack_type,    # targeted or untargeted
    attack_fn,      # attack function
    attack_kwargs,  # hyperparam
    save_path,
    target_cls=0,
):
    images, labels = next(iter(dataset["test_loader"]))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    # targeted
    if attack_type == "targeted":
        targets = torch.full_like(labels, target_cls)
        adv_images = attack_fn(model, images, targets, **attack_kwargs)

        with torch.no_grad():
            adv_preds = model(adv_images).argmax(dim=1)

        mask = (labels != target_cls) & (adv_preds == targets)
        if mask.sum().item() == 0:
            mask = labels != target_cls
    else:
        # untargeted
        adv_images = attack_fn(model, images, labels, **attack_kwargs)

        with torch.no_grad():
            adv_preds = model(adv_images).argmax(dim=1)

        mask = adv_preds != labels

    # no successful attack
    if mask.sum().item() == 0:
        mask = torch.arange(images.size(0), device=device) < 5

    save_path = save_attack_visualization(
        images[mask],
        adv_images[mask],
        preds[mask],
        adv_preds[mask],
        save_path,
        class_names=dataset["class_names"],
    )

    print(f"saved {attack_name} {attack_type}: {save_path}")

# run attacks and eval
def evaluate_attacks(
    dataset_name,
    model,
    dataset,
    *,
    device,
    results_dir,
    fgsm_eps,
    pgd_eps,
    pgd_eps_step,
    pgd_k,
    target_cls=0,
    num_samples=100,
):
    print()
    print(f"[{dataset_name}]")
    print(f"target cls: {target_cls}")

    fgsm_targeted_rate, fgsm_targeted_time = eval_attack_targeted(
        model,
        dataset["test_loader"],
        fgsm_targeted,
        device=device,
        target_cls=target_cls,
        num_samples=num_samples,
        desc=f"{dataset_name.lower()} fgsm targeted",
        eps=fgsm_eps,
    )
    print(f"fgsm targeted success rate: {fgsm_targeted_rate:.4f}")
    print(f"fgsm targeted time: {fgsm_targeted_time:.2f}s")
    save_attack_result(
        model,
        dataset,
        device=device,
        attack_name="fgsm",
        attack_type="targeted",
        attack_fn=fgsm_targeted,
        attack_kwargs={"eps": fgsm_eps},
        save_path=results_dir / f"{dataset_name.lower()}_fgsm_targeted.png",
        target_cls=target_cls,
    )

    pgd_targeted_rate, pgd_targeted_time = eval_attack_targeted(
        model,
        dataset["test_loader"],
        pgd_targeted,
        device=device,
        target_cls=target_cls,
        num_samples=num_samples,
        desc=f"{dataset_name.lower()} pgd targeted",
        k=pgd_k,
        eps=pgd_eps,
        eps_step=pgd_eps_step,
    )
    print(f"pgd targeted success rate: {pgd_targeted_rate:.4f}")
    print(f"pgd targeted time: {pgd_targeted_time:.2f}s")
    save_attack_result(
        model,
        dataset,
        device=device,
        attack_name="pgd",
        attack_type="targeted",
        attack_fn=pgd_targeted,
        attack_kwargs={
            "k": pgd_k,
            "eps": pgd_eps,
            "eps_step": pgd_eps_step,
        },
        save_path=results_dir / f"{dataset_name.lower()}_pgd_targeted.png",
        target_cls=target_cls,
    )

    fgsm_untargeted_rate, fgsm_untargeted_time = eval_attack_untargeted(
        model,
        dataset["test_loader"],
        fgsm_untargeted,
        device=device,
        num_samples=num_samples,
        desc=f"{dataset_name.lower()} fgsm untargeted",
        eps=fgsm_eps,
    )
    print(f"fgsm untargeted success rate: {fgsm_untargeted_rate:.4f}")
    print(f"fgsm untargeted time: {fgsm_untargeted_time:.2f}s")
    save_attack_result(
        model,
        dataset,
        device=device,
        attack_name="fgsm",
        attack_type="untargeted",
        attack_fn=fgsm_untargeted,
        attack_kwargs={"eps": fgsm_eps},
        save_path=results_dir / f"{dataset_name.lower()}_fgsm_untargeted.png",
    )

    pgd_untargeted_rate, pgd_untargeted_time = eval_attack_untargeted(
        model,
        dataset["test_loader"],
        pgd_untargeted,
        device=device,
        num_samples=num_samples,
        desc=f"{dataset_name.lower()} pgd untargeted",
        k=pgd_k,
        eps=pgd_eps,
        eps_step=pgd_eps_step,
    )
    print(f"pgd untargeted success rate: {pgd_untargeted_rate:.4f}")
    print(f"pgd untargeted time: {pgd_untargeted_time:.2f}s")
    save_attack_result(
        model,
        dataset,
        device=device,
        attack_name="pgd",
        attack_type="untargeted",
        attack_fn=pgd_untargeted,
        attack_kwargs={
            "k": pgd_k,
            "eps": pgd_eps,
            "eps_step": pgd_eps_step,
        },
        save_path=results_dir / f"{dataset_name.lower()}_pgd_untargeted.png",
    )

# train -> eval -> attack eval -> save visualization
def main():
    device = get_device()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())

    mnist = get_dataloaders("mnist", batch_size=64, data_root="data")
    cifar10 = get_dataloaders("cifar10", batch_size=64, data_root="data")

    mnist_model = MNISTCNN(
        use_dropout=False,
        use_bn=False,
        use_gelu=False,
        use_softmax=False,
    )
    mnist_model, mnist_history = train_classifier(
        mnist_model,
        mnist["train_loader"],
        mnist["test_loader"],
        device=device,
        epochs=3,
        lr=1e-3,
        run_name="MNIST",
    )
    print()
    print("[MNIST clean]")
    print(f"final train loss: {mnist_history['train_loss'][-1]:.4f}")
    print(f"final train acc: {mnist_history['train_acc'][-1]:.4f}")
    print(f"final clean test acc: {mnist_history['test_acc'][-1]:.4f}")

    # baseline: scratch ResNet-18 to avoid download dependency
    cifar_model = CIFARResNet(
        use_pretrained=False,
        use_gelu=False,
        use_softmax=False,
    )
    cifar_model, cifar_history = train_classifier(
        cifar_model,
        cifar10["train_loader"],
        cifar10["test_loader"],
        device=device,
        epochs=10,
        lr=1e-3,
        run_name="CIFAR-10",
    )
    print()
    print("[CIFAR-10 clean]")
    print(f"final train loss: {cifar_history['train_loss'][-1]:.4f}")
    print(f"final train acc: {cifar_history['train_acc'][-1]:.4f}")
    print(f"final clean test acc: {cifar_history['test_acc'][-1]:.4f}")

    evaluate_attacks(
        "mnist",
        mnist_model,
        mnist,
        device=device,
        results_dir=results_dir,
        fgsm_eps=0.3,
        pgd_eps=0.3,
        pgd_eps_step=0.01,
        pgd_k=40,
        target_cls=0,
        num_samples=100,
    )

    evaluate_attacks(
        "cifar10",
        cifar_model,
        cifar10,
        device=device,
        results_dir=results_dir,
        fgsm_eps=8 / 255,
        pgd_eps=8 / 255,
        pgd_eps_step=2 / 255,
        pgd_k=10,
        target_cls=0,
        num_samples=100,
    )


if __name__ == "__main__":
    main()
