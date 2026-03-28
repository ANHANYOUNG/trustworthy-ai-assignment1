import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted
from models.cifar_resnet import CIFARResNet
from models.mnist_cnn import MNISTCNN
from utils.data import get_dataloaders
from utils.eval_attack import eval_attack_targeted, eval_attack_untargeted

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def get_display_name(dataset_name):
    if dataset_name == "cifar10":
        return "CIFAR-10"
    return "MNIST"


def setup_paper_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )


def format_attack_name(attack_name, attack_type):
    return f"{attack_name.upper()} {attack_type}"


def get_sweep_settings(dataset_name):
    if dataset_name == "mnist":
        return {
            "batch_size": 64,
            "target_cls": 0,
            "num_samples": 500,
            "eps_list": [0.05, 0.10, 0.20, 0.30],
            "pgd_k": 40,
            "pgd_eps_step": 0.01,
        }

    return {
        "batch_size": 64,
        "target_cls": 0,
        "num_samples": 500,
        "eps_list": [2 / 255, 4 / 255, 8 / 255, 16 / 255],
        "pgd_k": 10,
        "pgd_eps_step": 2 / 255,
    }


def get_ablation_csv_path():
    candidates = [
        Path("results/ablation/ablation_results.csv"),
        Path("results/ablation_results.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("ablation_results.csv not found")

# load the best for eps sweep
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

# run attack
def evaluate_attack(dataset_name, model, dataset, attack_name, attack_type, eps, *, device):
    settings = get_sweep_settings(dataset_name)
    target_cls = settings["target_cls"]
    num_samples = settings["num_samples"]

    if attack_name == "fgsm":
        if attack_type == "targeted":
            success_rate, adv_test_acc, time = eval_attack_targeted(
                model,
                dataset["test_loader"],
                fgsm_targeted,
                device=device,
                target_cls=target_cls,
                num_samples=num_samples,
                desc=f"{dataset_name} fgsm targeted eps={eps:.4f}",
                eps=eps,
            )
            return success_rate, adv_test_acc, time, None

        success_rate, adv_test_acc, time = eval_attack_untargeted(
            model,
            dataset["test_loader"],
            fgsm_untargeted,
            device=device,
            num_samples=num_samples,
            desc=f"{dataset_name} fgsm untargeted eps={eps:.4f}",
            eps=eps,
        )
        return success_rate, adv_test_acc, time, None

    eps_step = settings["pgd_eps_step"]
    if attack_type == "targeted":
        success_rate, adv_test_acc, time = eval_attack_targeted(
            model,
            dataset["test_loader"],
            pgd_targeted,
            device=device,
            target_cls=target_cls,
            num_samples=num_samples,
            desc=f"{dataset_name} pgd targeted eps={eps:.4f}",
            k=settings["pgd_k"],
            eps=eps,
            eps_step=eps_step,
        )
        return success_rate, adv_test_acc, time, eps_step

    success_rate, adv_test_acc, time = eval_attack_untargeted(
        model,
        dataset["test_loader"],
        pgd_untargeted,
        device=device,
        num_samples=num_samples,
        desc=f"{dataset_name} pgd untargeted eps={eps:.4f}",
        k=settings["pgd_k"],
        eps=eps,
        eps_step=eps_step,
    )
    return success_rate, adv_test_acc, time, eps_step


def save_rows(rows, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with save_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_plot(dataset_name, rows, metric_key, ylabel, save_path):
    setup_paper_style()
    style_map = {
        "targeted": {"linestyle": "-", "marker": "o", "color": "#1f77b4"},
        "untargeted": {"linestyle": "--", "marker": "s", "color": "#d62728"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    for axis_idx, attack_name in enumerate(["fgsm", "pgd"]):
        ax = axes[axis_idx]
        attack_rows = [row for row in rows if row["attack_name"] == attack_name]

        for attack_type in ["targeted", "untargeted"]:
            type_rows = sorted(
                [row for row in attack_rows if row["attack_type"] == attack_type],
                key=lambda row: row["eps"],
            )
            style = style_map[attack_type]
            ax.plot(
                [row["eps"] for row in type_rows],
                [row[metric_key] for row in type_rows],
                label=attack_type.capitalize(),
                linewidth=1.8,
                markersize=5,
                **style,
            )

        ax.set_title(f"{PANEL_LABELS[axis_idx]} {attack_name.upper()}")
        ax.set_xlabel("Epsilon")
        if axis_idx == 0:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"{get_display_name(dataset_name)} | {ylabel} vs epsilon", y=0.99)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.92])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def collect_samples(dataset, *, device, targeted, target_cls, num_samples=3):
    images_list = []
    labels_list = []
    collected = 0

    for images, labels in dataset["test_loader"]:
        if targeted:
            mask = labels != target_cls
            images = images[mask]
            labels = labels[mask]

        if labels.numel() == 0:
            continue

        remaining = num_samples - collected
        images_list.append(images[:remaining])
        labels_list.append(labels[:remaining])
        collected += min(remaining, labels.size(0))

        if collected >= num_samples:
            break

    images = torch.cat(images_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)
    return images, labels


def run_attack_for_visualization(model, images, labels, attack_name, attack_type, eps, *, dataset_name):
    settings = get_sweep_settings(dataset_name)
    if attack_name == "fgsm":
        if attack_type == "targeted":
            targets = torch.full_like(labels, settings["target_cls"])
            adv_images = fgsm_targeted(model, images, targets, eps=eps)
        else:
            adv_images = fgsm_untargeted(model, images, labels, eps=eps)
        eps_step = None
    else:
        eps_step = settings["pgd_eps_step"]
        if attack_type == "targeted":
            targets = torch.full_like(labels, settings["target_cls"])
            adv_images = pgd_targeted(
                model,
                images,
                targets,
                k=settings["pgd_k"],
                eps=eps,
                eps_step=eps_step,
            )
        else:
            adv_images = pgd_untargeted(
                model,
                images,
                labels,
                k=settings["pgd_k"],
                eps=eps,
                eps_step=eps_step,
            )

    with torch.no_grad():
        preds = model(adv_images).argmax(dim=1)

    return adv_images, preds, eps_step


def show_image(ax, image):
    image = image.detach().cpu()
    if image.size(0) == 1:
        ax.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(image.permute(1, 2, 0).clamp(0.0, 1.0))
    ax.axis("off")


def save_epsilon_visualization(dataset_name, model, dataset, attack_name, attack_type, eps_list, *, config_tag, results_dir):
    setup_paper_style()
    settings = get_sweep_settings(dataset_name)
    class_names = dataset["class_names"]
    targeted = attack_type == "targeted"
    images, labels = collect_samples(
        dataset,
        device=next(model.parameters()).device,
        targeted=targeted,
        target_cls=settings["target_cls"],
        num_samples=3,
    )

    with torch.no_grad():
        orig_preds = model(images).argmax(dim=1)

    num_samples = images.size(0)
    fig, axes = plt.subplots(
        len(eps_list),
        num_samples * 3,
        figsize=(2.45 * num_samples * 3, 2.45 * len(eps_list)),
    )
    if len(eps_list) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, eps in enumerate(eps_list):
        adv_images, adv_preds, eps_step = run_attack_for_visualization(
            model,
            images,
            labels,
            attack_name,
            attack_type,
            eps,
            dataset_name=dataset_name,
        )

        for sample_idx in range(num_samples):
            ax_orig = axes[row_idx, sample_idx * 3]
            ax_adv = axes[row_idx, sample_idx * 3 + 1]
            ax_pert = axes[row_idx, sample_idx * 3 + 2]

            image = images[sample_idx]
            adv_image = adv_images[sample_idx]
            perturb = torch.clamp((adv_image - image) * 10.0 + 0.5, 0.0, 1.0)

            show_image(ax_orig, image)
            show_image(ax_adv, adv_image)
            show_image(ax_pert, perturb)

            gt_name = class_names[int(labels[sample_idx].item())]
            orig_pred_name = class_names[int(orig_preds[sample_idx].item())]
            adv_pred_name = class_names[int(adv_preds[sample_idx].item())]

            if row_idx == 0:
                ax_orig.set_title(f"Sample {sample_idx + 1}\nOriginal", fontsize=10, pad=8)
                ax_adv.set_title("Adversarial", fontsize=10, pad=8)
                ax_pert.set_title("Perturbation x10", fontsize=10, pad=8)

            ax_orig.text(0.5, -0.08, f"GT: {gt_name}", transform=ax_orig.transAxes, ha="center", va="top", fontsize=8)
            ax_adv.text(0.5, -0.08, f"pred: {adv_pred_name}", transform=ax_adv.transAxes, ha="center", va="top", fontsize=8)
            if targeted:
                target_name = class_names[settings["target_cls"]]
                ax_pert.text(0.5, -0.08, f"target: {target_name}", transform=ax_pert.transAxes, ha="center", va="top", fontsize=8)
            else:
                ax_pert.text(0.5, -0.08, f"clean pred: {orig_pred_name}", transform=ax_pert.transAxes, ha="center", va="top", fontsize=8)

            if sample_idx == 0:
                row_label = f"{PANEL_LABELS[row_idx]} ε={eps:.4f}"
                if eps_step is not None:
                    row_label += f"\nstep={eps_step:.4f}"
                ax_orig.text(
                    -0.4,
                    0.5,
                    row_label,
                    transform=ax_orig.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                    clip_on=False,
                )

    fig.suptitle(
        (
            f"{get_display_name(dataset_name)} | {format_attack_name(attack_name, attack_type)} | epsilon sweep\n"
            f"Best clean: {config_tag}"
        ),
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0.06, 0.04, 1.0, 0.9])

    save_path = results_dir / f"{dataset_name}_{attack_name}_{attack_type}_sweep.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    device = get_device()
    results_dir = Path("results") / "sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_rows = load_best_clean_rows()
    all_rows = []

    print("device:", device)
    for dataset_name in ["mnist", "cifar10"]:
        settings = get_sweep_settings(dataset_name)
        dataset = get_dataloaders(
            dataset_name,
            batch_size=settings["batch_size"],
            data_root="data",
        )

        best_row = best_rows[dataset_name]
        model = load_model_from_row(dataset_name, best_row, device=device)

        print()
        print(f"[{get_display_name(dataset_name)} sweep]")
        print(f"best clean config: {best_row['config_tag']}")
        print(f"clean test acc: {float(best_row['clean_test_acc']):.4f}")
        print(f"ckpt: {best_row['ckpt_path']}")

        for attack_name, attack_type in [
            ("fgsm", "targeted"),
            ("fgsm", "untargeted"),
            ("pgd", "targeted"),
            ("pgd", "untargeted"),
        ]:
            for eps in settings["eps_list"]:
                success_rate, adv_test_acc, time, eps_step = evaluate_attack(
                    dataset_name,
                    model,
                    dataset,
                    attack_name,
                    attack_type,
                    eps,
                    device=device,
                )
                all_rows.append(
                    {
                        "dataset": dataset_name,
                        "config_tag": best_row["config_tag"],
                        "clean_test_acc": float(best_row["clean_test_acc"]),
                        "attack_name": attack_name,
                        "attack_type": attack_type,
                        "eps": eps,
                        "eps_step": "" if eps_step is None else eps_step,
                        "success_rate": success_rate,
                        "adv_test_acc": adv_test_acc,
                        "time": time,
                        "target_cls": settings["target_cls"],
                        "num_samples": settings["num_samples"],
                        "pgd_k": settings["pgd_k"],
                        "ckpt_path": best_row["ckpt_path"],
                    }
                )

            save_epsilon_visualization(
                dataset_name,
                model,
                dataset,
                attack_name,
                attack_type,
                settings["eps_list"],
                config_tag=best_row["config_tag"],
                results_dir=results_dir,
            )

    save_rows(all_rows, results_dir / "epsilon_sweep_results.csv")

    for dataset_name in ["mnist", "cifar10"]:
        dataset_rows = [row for row in all_rows if row["dataset"] == dataset_name]
        save_metric_plot(
            dataset_name,
            dataset_rows,
            "success_rate",
            "success rate",
            results_dir / f"{dataset_name}_success_rate_vs_epsilon.png",
        )
        save_metric_plot(
            dataset_name,
            dataset_rows,
            "adv_test_acc",
            "adv test acc",
            results_dir / f"{dataset_name}_adv_test_acc_vs_epsilon.png",
        )

    print()
    print(f"saved: {results_dir / 'epsilon_sweep_results.csv'}")


if __name__ == "__main__":
    main()
