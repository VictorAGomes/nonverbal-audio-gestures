import numpy as np
import librosa
import torch
import torch.nn.functional as F
from config import IMG_SIZE


def _normalize_channel(arr):
    """Normaliza um array 2D para float32 no intervalo [0, 1]."""
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min + 1e-9)).astype(np.float32)


def _resize(rgb_hwc, target_size):
    """Redimensiona array (H, W, 3) float32 via interpolação bilinear.
    Usa torch.nn.functional.interpolate — sem dependência de PIL.
    """
    t = torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    t = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).numpy()                  # (H, W, 3)


def spectrogram_to_image(S_db, target_size=IMG_SIZE, representation='mel'):
    """Converte um Mel-espectrograma em dB para um array float32 (H, W, 3) em [0, 1].

    Representações disponíveis:
      'mel'   -> R=G=B=S_db  (apenas espectrograma, escala de cinza como RGB)
      'delta' -> R=S_db, G=delta, B=delta    (dinâmica de 1ª ordem)

    Cada canal é normalizado independentemente para [0, 1] em float32.
    Sem quantização para uint8 — precisão contínua preservada.
    """
    if representation == 'mel':
        ch  = _normalize_channel(S_db)
        rgb = np.stack([ch, ch, ch], axis=-1)

    elif representation == 'delta':
        delta = librosa.feature.delta(S_db)
        r = _normalize_channel(S_db)
        g = _normalize_channel(delta)
        rgb = np.stack([r, g, g], axis=-1)

    else:
        raise ValueError(
            f"representation deve ser 'mel' ou 'delta'. "
            f"Recebido: '{representation}'"
        )

    return _resize(rgb, target_size)
