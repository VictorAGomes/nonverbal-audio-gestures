import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from config import SR, N_FFT, HOP_LENGTH, N_MELS

# 1. Caminho do arquivo de áudio (.wav, .mp3, etc.)
audio_path = "WhatsApp Audio 2025-11-26 at 22.05.47.opus"  # troque pelo seu arquivo

# 2. Carregar o áudio (mono, 16 kHz)
y, sr = librosa.load(audio_path, sr=SR, mono=True)
print(f"Shape do sinal: {y.shape}, Sample rate: {sr}")

# 3. Calcular o mel-spectrograma

S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    fmin=0,
    fmax=sr // 2
)

# 4. Converter para dB (log)
S_db = librosa.power_to_db(S, ref=np.max)

print(f"Shape do mel-spectrograma: {S_db.shape}")

# 5. Plotar para conferir
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-spectrograma de teste")
plt.tight_layout()
plt.show()
