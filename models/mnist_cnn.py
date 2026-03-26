import torch.nn as nn


class MNISTCNN(nn.Module):
    # dropout, bn, softmax for comparison
    def __init__(
        self,
        *,
        use_dropout=False,
        use_bn=False,
        use_softmax=False,
        dropout_p=0.5,
        hidden_dim=128,
        num_cls=10,
    ):
        super().__init__()

        # input: (1, 28, 28)
        # 2 conv layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),        # (in_channels, out_channels, kernel_size, padding)
            nn.BatchNorm2d(32) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                       # (1, 28, 28) -> (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                       # (32, 14, 14) -> (64, 7, 7)
        )

        # 2 fc layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if use_dropout else nn.Identity(),
            nn.Linear(hidden_dim, num_cls),
        )

        self.out = nn.Softmax(dim=1) if use_softmax else nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.out(x)
        return x
