import torch.nn as nn


import torch.nn as nn

class NonVerbalCNN(nn.Module):
    """CNN customizada com 4 blocos convolucionais, BatchNorm e Global Average Pooling.
    Entrada: tensor RGB (3 x 128 x 128).
    GAP substitui o flatten, reduzindo de 16.384 para 256 features na camada FC.
    """
    def __init__(self, num_classes=3, arch_config=None):
        super(NonVerbalCNN, self).__init__()

        # Configuração padrão de segurança caso não seja passada nenhuma
        if arch_config is None:
            arch_config = {
                'kernel': (3, 3), 'global_pooling': 'average', 'dropout': 0.3
            }

        k_size = arch_config['kernel'][0]
        # Calcula automaticamente o padding 'same' para manter as dimensões
        pad = k_size // 2 

        self.conv_layers = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=k_size, padding=pad),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=k_size, padding=pad),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=k_size, padding=pad),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloco 4 (Mantendo os 4 blocos da sua original para estabilidade)
            nn.Conv2d(128, 256, kernel_size=k_size, padding=pad),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Configura o tipo de pooling final baseado no dicionário
        g_pool = arch_config.get('global_pooling', 'average')
        if g_pool == 'max':
            self.gap = nn.AdaptiveMaxPool2d(1)
            self.flat_size = 256
        elif g_pool == 'average':
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.flat_size = 256
        elif g_pool == 'flatten':
            self.gap = nn.Identity() # Não faz pooling, passa reto
            # 128x128 reduzido 4 vezes (divide por 2^4 = 16) fica 8x8. 
            # 256 canais * 8 * 8 = 16384
            self.flat_size = 16384 

        drop_rate = arch_config.get('dropout', 0.3)
        self.fc_layers = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.flat_size, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)
        return x
