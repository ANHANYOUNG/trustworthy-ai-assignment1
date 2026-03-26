from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    dataset_name,
    *,
    batch_size=128,
    data_root="data",
    num_workers=0,
    pin_memory=None,
    use_preprocess=False,
    download=True,
):
    dataset_name = dataset_name.lower()
    data_root = Path(data_root)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    if dataset_name == "mnist":
        train_transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()
        dataset_root = data_root / "mnist"
        input_shape = (1, 28, 28)

        train_dataset = datasets.MNIST(
            root=dataset_root,
            train=True,
            transform=train_transform,
            download=download,
        )
        test_dataset = datasets.MNIST(
            root=dataset_root,
            train=False,
            transform=test_transform,
            download=download,
        )
    else:
        dataset_name = "cifar10"
        dataset_root = data_root / "cifar10"

        if use_preprocess:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
            input_shape = (3, 224, 224)
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            test_transform = transforms.ToTensor()
            input_shape = (3, 32, 32)

        train_dataset = datasets.CIFAR10(
            root=dataset_root,
            train=True,
            transform=train_transform,
            download=download,
        )
        test_dataset = datasets.CIFAR10(
            root=dataset_root,
            train=False,
            transform=test_transform,
            download=download,
        )

    class_names = tuple(str(name) for name in train_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "dataset_name": dataset_name,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "num_classes": len(class_names),
        "input_shape": input_shape,
        "data_root": dataset_root,
        "use_preprocess": use_preprocess,
    }
