from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_attack_visualization(
    images,
    adv_images,
    preds,
    adv_preds,
    save_path,
    *,
    class_names=None,
    max_samples=5,
    perturb_scale=10.0,
):
    images = images.detach().cpu()
    adv_images = adv_images.detach().cpu()
    preds = torch.as_tensor(preds).detach().cpu()
    adv_preds = torch.as_tensor(adv_preds).detach().cpu()

    num_samples = min(
        max_samples,
        images.size(0),
        adv_images.size(0),
        preds.numel(),
        adv_preds.numel(),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 3)

    for i in range(num_samples):
        image = images[i]
        adv_image = adv_images[i]
        perturb = torch.clamp((adv_image - image) * perturb_scale + 0.5, 0.0, 1.0) # perturb visualization

        pred = int(preds[i])
        adv_pred = int(adv_preds[i])

        if class_names is not None:
            pred = class_names[pred]
            adv_pred = class_names[adv_pred]

        if image.shape[0] == 1:
            axes[i, 0].imshow(image.squeeze(0), cmap="gray")
            axes[i, 1].imshow(adv_image.squeeze(0), cmap="gray")
            axes[i, 2].imshow(perturb.squeeze(0), cmap="gray")
        else:
            axes[i, 0].imshow(image.permute(1, 2, 0))
            axes[i, 1].imshow(adv_image.permute(1, 2, 0))
            axes[i, 2].imshow(perturb.permute(1, 2, 0))

        axes[i, 0].set_title(f"original\npred: {pred}")
        axes[i, 1].set_title(f"adversarial\npred: {adv_pred}")
        axes[i, 2].set_title(f"perturbation x{perturb_scale:g}")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path
