import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Input image shape:
        # [batch, 3, 32, 32]

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = CNNClassifier()

    sample_batch = torch.rand(
        64,
        3,
        32,
        32
    )

    output = model(sample_batch)

    print(model)

    print("\nInput shape:")
    print(sample_batch.shape)

    print("\nOutput shape:")
    print(output.shape)