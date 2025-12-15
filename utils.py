import numpy as np
import librosa
import librosa.display
from PIL import Image
from config import SR, DURATION, N_FFT, HOP_LENGTH, N_MELS, IMG_SIZE, CLASSES

def spectrogram_to_image(S_db, target_size=IMG_SIZE):
    """
    Transforma o espectrograma em imagem RGB usando Matplotlib.
    Esta função é crítica para garantir que o input da inferência seja igual ao do treino.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    fig = Figure(figsize=(target_size[1] / 100, target_size[0] / 100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.axis('off')
    librosa.display.specshow(S_db, sr=SR, hop_length=HOP_LENGTH, ax=ax)
    fig.tight_layout(pad=0)
    
    canvas.draw()
    w, h = canvas.get_width_height()
    
    try:
        buf = canvas.tostring_rgb()
        img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    except AttributeError:
        buf = canvas.buffer_rgba()
        img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]
    
    pil = Image.fromarray(img_arr)
    pil = pil.resize((target_size[1], target_size[0]), Image.BICUBIC)
    return np.array(pil)