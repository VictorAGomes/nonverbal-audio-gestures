import torch.nn as nn


class NonVerbalCNN(nn.Module):
    """CNN customizada com 4 blocos convolucionais e BatchNorm.
    Entrada: imagem RGB 128x128 (3 x 128 x 128).
    """
    def __init__(self, num_classes=3):
        super(NonVerbalCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Bloco 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Após 4x MaxPool2d(2): 128 → 8  →  256 * 8 * 8 = 16.384
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
