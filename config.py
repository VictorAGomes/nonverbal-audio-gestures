"""
Configurações globais do projeto de reconhecimento de gestos não-verbais por áudio.
Centralizando todas as constantes para garantir consistência entre treino e inferência.
"""

# Parâmetros de áudio
SR = 16000              # Sample rate
DURATION = None         # Duração em segundos (None = usa duração total do áudio)

# Parâmetros de espectrograma
N_FFT = 1024            # Tamanho FFT
HOP_LENGTH = 256        # Passo entre janelas
N_MELS = 128            # Bandas Mel

# Parâmetros de imagem
IMG_SIZE = (128, 128)   # Tamanho final da imagem (altura, largura)

# Classes
CLASSES = ['assobio', 'dedo', 'palma']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
