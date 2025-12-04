import numpy as np
import librosa
import glob
import os
import matplotlib
matplotlib.use('Agg')  # backend sem interface gráfica
import matplotlib.pyplot as plt
import librosa.display

def load_and_normalize(path, sr=16000):
    # Carrega em mono
    y, sr = librosa.load(path, sr=sr, mono=True)
    
    # Normalização de ganho: escala para [-1, 1]
    max_val = np.max(np.abs(y)) + 1e-9
    y = y / max_val
    
    return y, sr

def random_time_shift(y, shift_max=0.2):
    """
    Desloca o áudio no tempo em até shift_max da duração (fração).
    """
    n = len(y)
    shift = int(np.random.uniform(-shift_max, shift_max) * n)
    return np.roll(y, shift)


def random_pitch_shift(y, sr, n_steps_range=(-2, 2)):
    """
    Variação aleatória de pitch em semitons.
    """
    n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    y_ps = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    return y_ps


def add_background_noise(y, noise_level_db_range=(-30, -15)):
    """
    Adiciona ruído branco com nível em dB relativo ao sinal.
    """
    # Energia RMS do sinal
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    
    # Escolhe nível de ruído em dB
    noise_db = np.random.uniform(noise_level_db_range[0], noise_level_db_range[1])
    noise_rms = rms * 10**(noise_db / 20.0)
    
    noise = np.random.normal(0, noise_rms, size=y.shape)
    y_noisy = y + noise
    # Re-normaliza para evitar clipping
    max_val = np.max(np.abs(y_noisy)) + 1e-9
    y_noisy = y_noisy / max_val
    return y_noisy

def augment_audio(y, sr, apply_time_shift=True, apply_pitch=True, apply_noise=True):
    """
    Aplica uma combinação de augmentations.
    """
    y_aug = y.copy()
    
    if apply_time_shift and np.random.rand() < 0.7:
        y_aug = random_time_shift(y_aug)
    
    if apply_pitch and np.random.rand() < 0.5:
        y_aug = random_pitch_shift(y_aug, sr)
    
    if apply_noise and np.random.rand() < 0.8:
        y_aug = add_background_noise(y_aug)
    
    # Garante normalização final
    max_val = np.max(np.abs(y_aug)) + 1e-9
    y_aug = y_aug / max_val
    return y_aug

files = glob.glob("data/*.opus")

out_dir = "augmented_specs"
img_dir = os.path.join(out_dir, "images")
os.makedirs(img_dir, exist_ok=True)

for path in files:
    y, sr = load_and_normalize(path, sr=16000)
    
    # Gera 3 versões augmentadas
    for i in range(3):
        y_aug = augment_audio(y, sr)
        # aqui você seguiria para extrair o mel-spectrograma
        # e alimentar o modelo / salvar como .npy ou imagem
        mel_spec = librosa.feature.melspectrogram(
            y=y_aug,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            fmin=0,
            fmax=sr // 2
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Salvar .npy
        base = os.path.splitext(os.path.basename(path))[0]
        npy_path = os.path.join(out_dir, f"{base}_aug{i}.npy")
        os.makedirs(out_dir, exist_ok=True)
        np.save(npy_path, mel_spec_db)
        
        # Salvar imagem do espectrograma
        img_path = os.path.join(img_dir, f"{base}_aug{i}.png")
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel', fmax=sr//2, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{base} - aug{i}")
        plt.tight_layout()
        plt.savefig(img_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
        plt.close()