import csv
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted
from models.cifar_resnet import CIFARResNet
from models.mnist_cnn import MNISTCNN
from utils.data import get_dataloaders
from utils.eval_attack import eval_attack_targeted, eval_attack_untargeted
from utils.train import eval_accuracy, train_classifier


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_eps(dataset_name, eps):
    if dataset_name == "cifar10":
        scaled = eps * 255
        rounded = round(scaled)
        if abs(scaled - rounded) < 1e-8:
            return f"{rounded}/255"
        return f"{scaled:.2f}/255"
    return f"{eps:.4f}"


def format_attack_settings(dataset_name, attack_settings):
    return (
        f"FGSM eps={format_eps(dataset_name, attack_settings['fgsm_eps'])} | "
        f"PGD eps={format_eps(dataset_name, attack_settings['pgd_eps'])}, "
        f"step={format_eps(dataset_name, attack_settings['pgd_eps_step'])}, "
        f"k={attack_settings['pgd_k']}"
    )


def get_configs():
    configs = []
    for use_dropout, use_bn, use_gelu in product([False, True], repeat=3):
        configs.append(
            {
                "use_dropout": use_dropout,
                "use_bn": use_bn,
                "use_gelu": use_gelu,
            }
        )
    return configs


def config_tag(config):
    return (
        f"d{int(config['use_dropout'])}"
        f"_b{int(config['use_bn'])}"
        f"_g{int(config['use_gelu'])}"
    )


def row_to_config(row):
    return {
        "use_dropout": bool(int(row["use_dropout"])),
        "use_bn": bool(int(row["use_bn"])),
        "use_gelu": bool(int(row["use_gelu"])),
    }


def format_config_text(row):
    return (
        f"{row['config_tag']} | "
        f"d={int(row['use_dropout'])}, "
        f"b={int(row['use_bn'])}, "
        f"g={int(row['use_gelu'])}"
    )


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


def get_train_settings(dataset_name):
    if dataset_name == "mnist":
        return {
            "batch_size": 64,
            "epochs": 3,
            "lr": 1e-3,
        }

    return {
        "batch_size": 64,
        "epochs": 10,
        "lr": 1e-3,
    }


def get_attack_settings(dataset_name):
    if dataset_name == "mnist":
        return {
            "target_cls": 0,
            "num_samples": 1000,
            "fgsm_eps": 0.3,
            "pgd_eps": 0.3,
            "pgd_eps_step": 0.01,
            "pgd_k": 40,
        }

    return {
        "target_cls": 0,
        "num_samples": 1000,
        "fgsm_eps": 8 / 255,
        "pgd_eps": 8 / 255,
        "pgd_eps_step": 2 / 255,
        "pgd_k": 10,
    }


def load_or_train_model(dataset_name, config, dataset, *, device, ckpt_root):
    ckpt_dir = ckpt_root / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{dataset_name}_{config_tag(config)}.pt"

    model = build_model(dataset_name, config)

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        history = checkpoint.get("history")
        print(f"load ckpt: {ckpt_path}")
    else:
        train_settings = get_train_settings(dataset_name)
        print(f"train model: {dataset_name} {config_tag(config)}")
        model, history = train_classifier(
            model,
            dataset["train_loader"],
            dataset["test_loader"],
            device=device,
            epochs=train_settings["epochs"],
            lr=train_settings["lr"],
            run_name=f"{dataset_name} {config_tag(config)}",
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "config": config,
            },
            ckpt_path,
        )
        print(f"save ckpt: {ckpt_path}")

    clean_test_acc = eval_accuracy(
        model,
        dataset["test_loader"],
        device=device,
        desc=f"{dataset_name} {config_tag(config)} clean eval",
    )

    return model, clean_test_acc, ckpt_path


def evaluate_attacks(dataset_name, model, dataset, config, *, device):
    attack_settings = get_attack_settings(dataset_name)
    tag = config_tag(config)

    fgsm_targeted_sr, fgsm_targeted_adv_acc, fgsm_targeted_time = eval_attack_targeted(
        model,
        dataset["test_loader"],
        fgsm_targeted,
        device=device,
        target_cls=attack_settings["target_cls"],
        num_samples=attack_settings["num_samples"],
        desc=f"{dataset_name} {tag} fgsm targeted",
        eps=attack_settings["fgsm_eps"],
    )
    fgsm_untargeted_sr, fgsm_untargeted_adv_acc, fgsm_untargeted_time = eval_attack_untargeted(
        model,
        dataset["test_loader"],
        fgsm_untargeted,
        device=device,
        num_samples=attack_settings["num_samples"],
        desc=f"{dataset_name} {tag} fgsm untargeted",
        eps=attack_settings["fgsm_eps"],
    )
    pgd_targeted_sr, pgd_targeted_adv_acc, pgd_targeted_time = eval_attack_targeted(
        model,
        dataset["test_loader"],
        pgd_targeted,
        device=device,
        target_cls=attack_settings["target_cls"],
        num_samples=attack_settings["num_samples"],
        desc=f"{dataset_name} {tag} pgd targeted",
        k=attack_settings["pgd_k"],
        eps=attack_settings["pgd_eps"],
        eps_step=attack_settings["pgd_eps_step"],
    )
    pgd_untargeted_sr, pgd_untargeted_adv_acc, pgd_untargeted_time = eval_attack_untargeted(
        model,
        dataset["test_loader"],
        pgd_untargeted,
        device=device,
        num_samples=attack_settings["num_samples"],
        desc=f"{dataset_name} {tag} pgd untargeted",
        k=attack_settings["pgd_k"],
        eps=attack_settings["pgd_eps"],
        eps_step=attack_settings["pgd_eps_step"],
    )

    return {
        "target_cls": attack_settings["target_cls"],
        "num_samples": attack_settings["num_samples"],
        "fgsm_eps": attack_settings["fgsm_eps"],
        "pgd_eps": attack_settings["pgd_eps"],
        "pgd_eps_step": attack_settings["pgd_eps_step"],
        "pgd_k": attack_settings["pgd_k"],
        "fgsm_targeted_sr": fgsm_targeted_sr,
        "fgsm_targeted_adv_acc": fgsm_targeted_adv_acc,
        "fgsm_targeted_time": fgsm_targeted_time,
        "fgsm_untargeted_sr": fgsm_untargeted_sr,
        "fgsm_untargeted_adv_acc": fgsm_untargeted_adv_acc,
        "fgsm_untargeted_time": fgsm_untargeted_time,
        "pgd_targeted_sr": pgd_targeted_sr,
        "pgd_targeted_adv_acc": pgd_targeted_adv_acc,
        "pgd_targeted_time": pgd_targeted_time,
        "pgd_untargeted_sr": pgd_untargeted_sr,
        "pgd_untargeted_adv_acc": pgd_untargeted_adv_acc,
        "pgd_untargeted_time": pgd_untargeted_time,
    }


def save_rows(rows, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with save_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_display_name(dataset_name):
    if dataset_name == "cifar10":
        return "CIFAR-10"
    return "MNIST"


def get_representative_rows(rows):
    best_attack_row = max(rows, key=lambda row: row["avg_attack_success_rate"])
    best_robust_row = min(rows, key=lambda row: row["avg_attack_success_rate"])

    return [
        ("best attack", best_attack_row),
        ("best robust", best_robust_row),
    ]


def load_model_from_row(dataset_name, row, *, device):
    model = build_model(dataset_name, row_to_config(row))
    checkpoint = torch.load(row["ckpt_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def collect_visualization_samples(
    dataset,
    *,
    device,
    num_samples=5,
    target_cls=0,
    targeted=False,
):
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


def run_attack_mode(model, images, labels, mode, attack_settings):
    if mode == "clean":
        vis_images = images
    elif mode == "fgsm_targeted":
        targets = torch.full_like(labels, attack_settings["target_cls"])
        vis_images = fgsm_targeted(
            model,
            images,
            targets,
            eps=attack_settings["fgsm_eps"],
        )
    elif mode == "fgsm_untargeted":
        vis_images = fgsm_untargeted(
            model,
            images,
            labels,
            eps=attack_settings["fgsm_eps"],
        )
    elif mode == "pgd_targeted":
        targets = torch.full_like(labels, attack_settings["target_cls"])
        vis_images = pgd_targeted(
            model,
            images,
            targets,
            k=attack_settings["pgd_k"],
            eps=attack_settings["pgd_eps"],
            eps_step=attack_settings["pgd_eps_step"],
        )
    else:
        vis_images = pgd_untargeted(
            model,
            images,
            labels,
            k=attack_settings["pgd_k"],
            eps=attack_settings["pgd_eps"],
            eps_step=attack_settings["pgd_eps_step"],
        )

    with torch.no_grad():
        preds = model(vis_images).argmax(dim=1)

    return vis_images, preds


def show_image(ax, image):
    image = image.detach().cpu()
    if image.size(0) == 1:
        ax.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(image.permute(1, 2, 0).clamp(0.0, 1.0))
    ax.axis("off")


def save_prediction_visualizations(dataset_name, dataset, rows, *, results_dir, device):
    attack_settings = get_attack_settings(dataset_name)
    display_name = get_display_name(dataset_name)
    representative_rows = get_representative_rows(rows)
    class_names = dataset["class_names"]
    modes = [
        "clean",
        "fgsm_targeted",
        "fgsm_untargeted",
        "pgd_targeted",
        "pgd_untargeted",
    ]

    print()
    print(f"[{display_name} prediction visualization]")
    for label, row in representative_rows:
        print(f"{label}: {format_config_text(row)}")

    for mode in modes:
        targeted = mode in {"fgsm_targeted", "pgd_targeted"}
        images, labels = collect_visualization_samples(
            dataset,
            device=device,
            num_samples=5,
            target_cls=attack_settings["target_cls"],
            targeted=targeted,
        )

        fig, axes = plt.subplots(
            len(representative_rows),
            images.size(0),
            figsize=(3.0 * images.size(0), 3.0 * len(representative_rows)),
        )

        if len(representative_rows) == 1:
            axes = [axes]
        if images.size(0) == 1:
            axes = [[ax] for ax in axes]

        for row_idx, (label, row) in enumerate(representative_rows):
            model = load_model_from_row(dataset_name, row, device=device)
            vis_images, preds = run_attack_mode(model, images, labels, mode, attack_settings)

            for col_idx in range(images.size(0)):
                ax = axes[row_idx][col_idx]
                show_image(ax, vis_images[col_idx])

                gt_name = class_names[int(labels[col_idx].item())]
                pred_name = class_names[int(preds[col_idx].item())]
                if targeted:
                    target_name = class_names[attack_settings["target_cls"]]
                    ax.set_title(
                        f"gt:{gt_name}\npred:{pred_name}\ntgt:{target_name}",
                        fontsize=9,
                    )
                else:
                    ax.set_title(
                        f"gt:{gt_name}\npred:{pred_name}",
                        fontsize=9,
                    )

                if col_idx == 0:
                    row_label = (
                        f"{label}\n"
                        f"{row['config_tag']}\n"
                        f"d={int(row['use_dropout'])}, "
                        f"b={int(row['use_bn'])}, "
                        f"g={int(row['use_gelu'])}"
                    )
                    ax.text(
                        -0.72,
                        0.5,
                        row_label,
                        transform=ax.transAxes,
                        fontsize=9,
                        va="center",
                        ha="left",
                        clip_on=False,
                    )

        fig.suptitle(
            (
                f"{display_name} | {mode.replace('_', ' ')} | prediction comparison\n"
                f"{format_attack_settings(dataset_name, attack_settings)}"
            ),
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout(rect=[0.24, 0.02, 1.0, 0.94])

        save_path = results_dir / f"{dataset_name}_ablation_{mode}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {save_path}")


def print_dataset_summary(dataset_name, rows):
    print()
    print(f"[{dataset_name.upper()} ablation]")
    print(
        f"{'config':<10}"
        f"{'dropout':<10}"
        f"{'bn':<6}"
        f"{'gelu':<6}"
        f"{'clean_acc':<12}"
        f"{'avg_sr':<10}"
    )
    print("-" * 54)

    for row in rows:
        print(
            f"{row['config_tag']:<10}"
            f"{row['use_dropout']:<10}"
            f"{row['use_bn']:<6}"
            f"{row['use_gelu']:<6}"
            f"{row['clean_test_acc']:<12.4f}"
            f"{row['avg_attack_success_rate']:<10.4f}"
        )


def main():
    device = get_device()
    ckpt_root = Path("ckpts")
    results_dir = Path("results") / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())

    configs = get_configs()
    all_rows = []

    for dataset_name in ["mnist", "cifar10"]:
        train_settings = get_train_settings(dataset_name)
        dataset = get_dataloaders(
            dataset_name,
            batch_size=train_settings["batch_size"],
            data_root="data",
        )

        dataset_rows = []
        for config in configs:
            model, clean_test_acc, ckpt_path = load_or_train_model(
                dataset_name,
                config,
                dataset,
                device=device,
                ckpt_root=ckpt_root,
            )
            attack_results = evaluate_attacks(
                dataset_name,
                model,
                dataset,
                config,
                device=device,
            )

            avg_attack_success_rate = (
                attack_results["fgsm_targeted_sr"]
                + attack_results["fgsm_untargeted_sr"]
                + attack_results["pgd_targeted_sr"]
                + attack_results["pgd_untargeted_sr"]
            ) / 4.0
            avg_attack_adv_acc = (
                attack_results["fgsm_targeted_adv_acc"]
                + attack_results["fgsm_untargeted_adv_acc"]
                + attack_results["pgd_targeted_adv_acc"]
                + attack_results["pgd_untargeted_adv_acc"]
            ) / 4.0

            row = {
                "dataset": dataset_name,
                "config_tag": config_tag(config),
                "use_dropout": int(config["use_dropout"]),
                "use_bn": int(config["use_bn"]),
                "use_gelu": int(config["use_gelu"]),
                "clean_test_acc": clean_test_acc,
                "avg_attack_success_rate": avg_attack_success_rate,
                "avg_attack_adv_acc": avg_attack_adv_acc,
                "target_cls": attack_results["target_cls"],
                "num_samples": attack_results["num_samples"],
                "fgsm_eps": attack_results["fgsm_eps"],
                "fgsm_eps_label": format_eps(dataset_name, attack_results["fgsm_eps"]),
                "pgd_eps": attack_results["pgd_eps"],
                "pgd_eps_label": format_eps(dataset_name, attack_results["pgd_eps"]),
                "pgd_eps_step": attack_results["pgd_eps_step"],
                "pgd_eps_step_label": format_eps(dataset_name, attack_results["pgd_eps_step"]),
                "pgd_k": attack_results["pgd_k"],
                "fgsm_targeted_sr": attack_results["fgsm_targeted_sr"],
                "fgsm_targeted_adv_acc": attack_results["fgsm_targeted_adv_acc"],
                "fgsm_targeted_time": attack_results["fgsm_targeted_time"],
                "fgsm_untargeted_sr": attack_results["fgsm_untargeted_sr"],
                "fgsm_untargeted_adv_acc": attack_results["fgsm_untargeted_adv_acc"],
                "fgsm_untargeted_time": attack_results["fgsm_untargeted_time"],
                "pgd_targeted_sr": attack_results["pgd_targeted_sr"],
                "pgd_targeted_adv_acc": attack_results["pgd_targeted_adv_acc"],
                "pgd_targeted_time": attack_results["pgd_targeted_time"],
                "pgd_untargeted_sr": attack_results["pgd_untargeted_sr"],
                "pgd_untargeted_adv_acc": attack_results["pgd_untargeted_adv_acc"],
                "pgd_untargeted_time": attack_results["pgd_untargeted_time"],
                "ckpt_path": str(ckpt_path),
            }
            dataset_rows.append(row)
            all_rows.append(row)

        print_dataset_summary(dataset_name, dataset_rows)
        save_prediction_visualizations(
            dataset_name,
            dataset,
            dataset_rows,
            results_dir=results_dir,
            device=device,
        )

    save_rows(all_rows, results_dir / "ablation_results.csv")
    print()
    print(f"saved: {results_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
