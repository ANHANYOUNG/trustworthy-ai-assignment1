import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def replace_relu(module, *, use_gelu):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU() if use_gelu else nn.ReLU(inplace=True))
        else:
            replace_relu(child, use_gelu=use_gelu)


class CIFARResNet(nn.Module):
    def __init__(
        self,
        *,
        use_pretrained=False,
        use_gelu=False,
        use_softmax=False,
        num_cls=10,
    ):
        super().__init__()

        self.use_pretrained = use_pretrained

        # pretrained : ResNet-18 trained on ImageNet
        # input      : (64, 3, 224, 224)
        # normalize  : (64, 3, 224, 224)
        # conv1      : (64, 64, 112, 112)
        # bn1/relu   : (64, 64, 112, 112)
        # maxpool    : (64, 64, 56, 56)
        # layer1     : (64, 64, 56, 56)
        # layer2     : (64, 128, 28, 28)
        # layer3     : (64, 256, 14, 14)
        # layer4     : (64, 512, 7, 7)
        # avgpool    : (64, 512, 1, 1)
        # flatten    : (64, 512)
        # fc         : (64, 10)
        if use_pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

            # preprocess for pretrained on ImageNet
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )
        else:
            # scratch    : ResNet-18 without pretrained
            # input      : (64, 3, 32, 32)
            # conv1      : (64, 64, 32, 32)
            # bn1/relu   : (64, 64, 32, 32)
            # identity   : (64, 64, 32, 32)
            # layer1     : (64, 64, 32, 32)
            # layer2     : (64, 128, 16, 16)
            # layer3     : (64, 256, 8, 8)
            # layer4     : (64, 512, 4, 4)
            # avgpool    : (64, 512, 1, 1)
            # flatten    : (64, 512)
            # fc         : (64, 10)
            self.backbone = resnet18(weights=None)

            # change first conv for CIFAR-10 (3x32x32)
            # kernel_size 7 -> 3, stride 2 -> 1, padding 3 -> 1
            self.backbone.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            # not using maxpool (identity: y = x)
            self.backbone.maxpool = nn.Identity()

            self.mean = None
            self.std = None

        replace_relu(self.backbone, use_gelu=use_gelu)

        input_features = self.backbone.fc.in_features # 512 for ResNet-18
        self.backbone.fc = nn.Linear(input_features, num_cls)

        self.out = nn.Softmax(dim=1) if use_softmax else nn.Identity()

    def forward(self, x):
        if self.use_pretrained:
            x = (x - self.mean) / self.std

        x = self.backbone(x)
        x = self.out(x)
        return x

