import numpy as np
import librosa
import librosa.display
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from config import SR, HOP_LENGTH, IMG_SIZE


def spectrogram_to_image(S_db, target_size=IMG_SIZE):
    """Renderiza o espectrograma para uma imagem RGB de tamanho target_size.
    Compatível com diferentes backends matplotlib no macOS.
    
    Args:
        S_db: Mel-spectrograma em dB (array 2D)
        target_size: Tupla (altura, largura) para o tamanho final da imagem
        
    Returns:
        Array numpy com shape (altura, largura, 3) representando a imagem RGB
    """
    fig = Figure(figsize=(target_size[1] / 100, target_size[0] / 100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.axis('off')
    librosa.display.specshow(S_db, sr=SR, hop_length=HOP_LENGTH, ax=ax)
    fig.tight_layout(pad=0)

    canvas.draw()
    w, h = canvas.get_width_height()

    # Tenta diferentes APIs de buffer conforme disponível
    try:
        buf = canvas.tostring_rgb()
        img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    except AttributeError:
        try:
            buf = canvas.buffer_rgba()
            img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
        except Exception:
            buf = canvas.tostring_argb()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            img_arr = arr[..., 1:4]

    pil = Image.fromarray(img_arr)
    pil = pil.resize((target_size[1], target_size[0]), Image.BICUBIC)
    return np.array(pil)

