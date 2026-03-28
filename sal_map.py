import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted
from models.cifar_resnet import CIFARResNet
from models.mnist_cnn import MNISTCNN
from utils.data import get_dataloaders

TARGET_CLS = 0
PANEL_LABELS = ["(a)", "(b)", "(c)"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_display_name(dataset_name):
    if dataset_name == "cifar10":
        return "CIFAR-10"
    return "MNIST"


def setup_paper_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "figure.titlesize": 13,
        }
    )


def format_variant_name(name):
    if "targeted" in name:
        if "un" in name:
            return "Untargeted"
        return "Targeted"
    return "Clean"


def get_attack_settings(dataset_name):
    if dataset_name == "mnist":
        return {
            "fgsm_eps": 0.3,
            "pgd_eps": 0.3,
            "pgd_eps_step": 0.01,
            "pgd_k": 40,
        }

    return {
        "fgsm_eps": 8 / 255,
        "pgd_eps": 8 / 255,
        "pgd_eps_step": 2 / 255,
        "pgd_k": 10,
    }


def row_to_config(row):
    return {
        "use_dropout": bool(int(row["use_dropout"])),
        "use_bn": bool(int(row["use_bn"])),
        "use_gelu": bool(int(row["use_gelu"])),
    }


def build_model(dataset_name, config):
    if dataset_name == "mnist":
        return MNISTCNN(
            use_dropout=config["use_dropout"],
            use_bn=config["use_bn"],
            use_gelu=config["use_gelu"],
            use_softmax=False,
        )

    return CIFARResNet(
        use_pretrained=False,
        use_dropout=config["use_dropout"],
        use_bn=config["use_bn"],
        use_gelu=config["use_gelu"],
        use_softmax=False,
    )


def get_ablation_csv_path():
    candidates = [
        Path("results/ablation/ablation_results.csv"),
        Path("results/ablation_results.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("ablation_results.csv not found")


def load_best_clean_rows():
    csv_path = get_ablation_csv_path()
    rows = list(csv.DictReader(csv_path.open()))

    best_rows = {}
    for dataset_name in ["mnist", "cifar10"]:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        best_rows[dataset_name] = max(
            dataset_rows,
            key=lambda row: float(row["clean_test_acc"]),
        )
    return best_rows


def load_model_from_row(dataset_name, row, *, device):
    model = build_model(dataset_name, row_to_config(row))
    checkpoint = torch.load(row["ckpt_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def collect_correct_samples(model, dataset, *, device, num_samples=5, target_cls=TARGET_CLS):
    images_list = []
    labels_list = []
    preds_list = []
    collected = 0

    with torch.no_grad():
        for images, labels in dataset["test_loader"]:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            mask = (preds == labels) & (labels != target_cls)

            if mask.sum().item() == 0:
                continue

            images = images[mask]
            labels = labels[mask]
            preds = preds[mask]

            remaining = num_samples - collected
            images_list.append(images[:remaining])
            labels_list.append(labels[:remaining])
            preds_list.append(preds[:remaining])
            collected += min(remaining, labels.size(0))

            if collected >= num_samples:
                break

    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    preds = torch.cat(preds_list, dim=0)
    return images, labels, preds, dataset["class_names"]


def compute_saliency(model, images):
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    preds = logits.argmax(dim=1) # get the largest logit class
    selected = logits.gather(1, preds.unsqueeze(1)).sum()
    model.zero_grad()
    selected.backward() # backprop to get grad for each pixel for chosen class

    # large grad means small change can cause large change in logit
    saliency = x.grad.detach().abs()
    # mnist
    if saliency.size(1) == 1:
        saliency = saliency.squeeze(1)
    else:
        # cifar10
        saliency = saliency.max(dim=1).values

    return preds.detach(), saliency


def build_attack_variants(model, dataset_name, images, labels, *, target_cls=TARGET_CLS):
    settings = get_attack_settings(dataset_name)
    target = torch.full_like(labels, target_cls)

    fgsm_images = fgsm_untargeted(
        model,
        images,
        labels,
        eps=settings["fgsm_eps"],
    )
    fgsm_target_images = fgsm_targeted(
        model,
        images,
        target,
        eps=settings["fgsm_eps"],
    )
    pgd_images = pgd_untargeted(
        model,
        images,
        labels,
        k=settings["pgd_k"],
        eps=settings["pgd_eps"],
        eps_step=settings["pgd_eps_step"],
    )
    pgd_target_images = pgd_targeted(
        model,
        images,
        target,
        k=settings["pgd_k"],
        eps=settings["pgd_eps"],
        eps_step=settings["pgd_eps_step"],
    )

    return {
        "fgsm": [
            ("clean", images),
            ("fgsm targeted", fgsm_target_images),
            ("fgsm untargeted", fgsm_images),
        ],
        "pgd": [
            ("clean", images),
            ("pgd targeted", pgd_target_images),
            ("pgd untargeted", pgd_images),
        ],
    }


def show_image(ax, image):
    image = image.detach().cpu()
    if image.size(0) == 1:
        ax.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(image.permute(1, 2, 0).clamp(0.0, 1.0))
    ax.axis("off")


def show_saliency(ax, saliency):
    ax.imshow(saliency.detach().cpu(), cmap="inferno")
    ax.axis("off")


def save_saliency_plot(dataset_name, model, row, labels, class_names, *, results_dir, attack_name, variants):
    setup_paper_style()
    images = variants[0][1]
    num_rows = len(variants)
    num_samples = images.size(0)
    fig, axes = plt.subplots(
        num_rows,
        num_samples * 2,
        figsize=(2.4 * num_samples * 2, 2.5 * num_rows),
    )
    if num_rows == 1:
        axes = axes.reshape(1, num_samples * 2)

    for variant_idx, (variant_name, variant_images) in enumerate(variants):
        preds, saliency = compute_saliency(model, variant_images)

        for sample_idx in range(num_samples):
            image_ax = axes[variant_idx, 2 * sample_idx]
            saliency_ax = axes[variant_idx, 2 * sample_idx + 1]

            show_image(image_ax, variant_images[sample_idx])
            show_saliency(saliency_ax, saliency[sample_idx])

            gt_name = class_names[int(labels[sample_idx].item())]
            pred_name = class_names[int(preds[sample_idx].item())]
            image_ax.text(
                0.5,
                -0.08,
                f"pred: {pred_name}",
                transform=image_ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )

            if variant_idx == 0:
                image_ax.set_title(f"Sample {sample_idx + 1}\nGT: {gt_name}", fontsize=10, pad=8)
                saliency_ax.set_title("Saliency", fontsize=10, pad=8)

        axes[variant_idx, 0].text(
            -0.22,
            0.5,
            f"{PANEL_LABELS[variant_idx]} {format_variant_name(variant_name)}",
            transform=axes[variant_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
            clip_on=False,
        )

    settings = get_attack_settings(dataset_name)
    if attack_name == "fgsm":
        attack_text = f"FGSM eps={settings['fgsm_eps']:.4f}"
    else:
        attack_text = (
            f"PGD eps={settings['pgd_eps']:.4f}, "
            f"step={settings['pgd_eps_step']:.4f}, k={settings['pgd_k']}"
        )

    fig.suptitle(
        (
            f"{get_display_name(dataset_name)} | {attack_name.upper()} saliency map | clean vs targeted vs untargeted\n"
            f"Best clean: {row['config_tag']} | target class: {TARGET_CLS} | {attack_text}"
        ),
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0.05, 0.02, 1.0, 0.9])

    save_path = results_dir / f"{dataset_name}_saliency_{attack_name}_targeted_vs_untargeted_{row['config_tag']}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def main():
    device = get_device()
    results_dir = Path("results") / "saliency_map"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_rows = load_best_clean_rows()
    print("device:", device)

    for dataset_name in ["mnist", "cifar10"]:
        dataset = get_dataloaders(dataset_name, batch_size=64, data_root="data")
        row = best_rows[dataset_name]
        model = load_model_from_row(dataset_name, row, device=device)
        images, labels, _, class_names = collect_correct_samples(
            model,
            dataset,
            device=device,
            num_samples=5,
            target_cls=TARGET_CLS,
        )
        attack_variants = build_attack_variants(
            model,
            dataset_name,
            images,
            labels,
            target_cls=TARGET_CLS,
        )

        print()
        print(f"[{get_display_name(dataset_name)} saliency]")
        print(f"best clean config: {row['config_tag']}")
        print(f"clean test acc: {float(row['clean_test_acc']):.4f}")
        print(f"target cls: {TARGET_CLS}")

        for attack_name in ["fgsm", "pgd"]:
            save_saliency_plot(
                dataset_name,
                model,
                row,
                labels,
                class_names,
                results_dir=results_dir,
                attack_name=attack_name,
                variants=attack_variants[attack_name],
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
