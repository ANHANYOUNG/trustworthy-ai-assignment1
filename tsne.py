import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

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
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
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


def extract_features(model, dataset_name, x):
    if dataset_name == "mnist":
        x = model.features(x)
        x = model.classifier[0](x) # flatten
        x = model.classifier[1](x) # linear (64*7*7 -> hidden_dim)
        x = model.classifier[2](x) # relu
        x = model.classifier[3](x) # dropout or identity
        x = model.classifier[4](x) # linear (hidden_dim -> num_cls)
        return x

    if model.use_pretrained:
        x = (x - model.mean) / model.std

    backbone = model.backbone
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    x = backbone.avgpool(x)
    x = torch.flatten(x, 1)
    return x


def collect_correct_samples(model, dataset, *, device, max_samples=1000, target_cls=TARGET_CLS):
    images_list = []
    labels_list = []
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

            remaining = max_samples - collected
            images_list.append(images[:remaining])
            labels_list.append(labels[:remaining])
            collected += min(remaining, labels.size(0))

            if collected >= max_samples:
                break

    return torch.cat(images_list, dim=0), torch.cat(labels_list, dim=0)


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


def save_tsne_plot(dataset_name, model, row, labels, class_names, *, results_dir, attack_name, variants):
    setup_paper_style()
    feature_blocks = []
    state_blocks = []
    label_blocks = []
    for state_idx, (_, state_images) in enumerate(variants):
        features = extract_features(model, dataset_name, state_images)
        feature_blocks.append(features.detach().cpu())
        state_blocks.append(torch.full((features.size(0),), state_idx, dtype=torch.long))
        label_blocks.append(labels.detach().cpu())

    all_features = torch.cat(feature_blocks, dim=0).numpy()
    all_states = torch.cat(state_blocks, dim=0).numpy()
    all_labels = torch.cat(label_blocks, dim=0).numpy()

    # tSNE hyperparam
    perplexity = min(30, max(5, (all_features.shape[0] - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(all_features) # run tSNE

    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

    fig, axes = plt.subplots(1, len(variants), figsize=(12.5, 4.2), sharex=True, sharey=True)
    if len(variants) == 1:
        axes = [axes]

    for state_idx, (title, _) in enumerate(variants):
        ax = axes[state_idx]
        mask = all_states == state_idx
        scatter = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=all_labels[mask],
            cmap="tab10",
            s=7,
            alpha=0.85,
            linewidths=0,
        )
        ax.set_title(f"{PANEL_LABELS[state_idx]} {format_variant_name(title)}")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("t-SNE 1")
        if state_idx == 0:
            ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.15, linewidth=0.5)
        ax.set_axisbelow(True)

    handles, _ = scatter.legend_elements(num=len(class_names))
    fig.legend(
        handles,
        class_names,
        title="Class",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(class_names), 5),
        frameon=False,
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
            f"{get_display_name(dataset_name)} | {attack_name.upper()} | clean vs targeted vs untargeted\n"
            f"Best clean: {row['config_tag']} | target class: {TARGET_CLS} | {attack_text}"
        ),
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.9])

    save_path = results_dir / f"{dataset_name}_tsne_{attack_name}_targeted_vs_untargeted_{row['config_tag']}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def main():
    device = get_device()
    results_dir = Path("results") / "t-SNE"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_rows = load_best_clean_rows()
    print("device:", device)

    for dataset_name in ["mnist", "cifar10"]:
        dataset = get_dataloaders(dataset_name, batch_size=64, data_root="data")
        row = best_rows[dataset_name]
        model = load_model_from_row(dataset_name, row, device=device)
        class_names = dataset["class_names"]
        images, labels = collect_correct_samples(
            model,
            dataset,
            device=device,
            max_samples=1000,
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
        print(f"[{get_display_name(dataset_name)} t-SNE]")
        print(f"best clean config: {row['config_tag']}")
        print(f"clean test acc: {float(row['clean_test_acc']):.4f}")
        print(f"target cls: {TARGET_CLS}")

        for attack_name in ["fgsm", "pgd"]:
            save_tsne_plot(
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
