import tkinter as tk
from tkinter import font
import sounddevice as sd
import numpy as np
import torch
import librosa
import threading

from model import NonVerbalCNN
import utils 

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecedor de Sons")
        self.root.geometry("400x400")
        self.root.configure(bg="#1e272e")

        self.recording = False
        self.audio_buffer = []
        self.stream = None
        
        # Carregar Modelo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

        # --- INTERFACE ---
        self.setup_ui()

    def load_model(self):
        print("Carregando modelo...")
        try:
            # Instancia usando a classe importada de model.py
            self.model = NonVerbalCNN(num_classes=len(utils.CLASSES))
            
            # Carrega os pesos
            self.model.load_state_dict(torch.load('best_nonverbal_model.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            self.model = None

    def setup_ui(self):
        self.lbl_status = tk.Label(self.root, text="Segure o botão para falar", 
                                   bg="#1e272e", fg="#d2dae2", font=("Helvetica", 12))
        self.lbl_status.pack(pady=30)

        # Botão Principal (Push-to-Talk)
        self.btn_rec = tk.Button(self.root, text="🎙️", 
                                 font=("Arial", 40), 
                                 bg="#ff5e57", fg="white", 
                                 activebackground="#ff3f34",
                                 bd=0, highlightthickness=0,
                                 width=3, height=1)
        self.btn_rec.pack(pady=10)
        
        # Bindings (Eventos de apertar e soltar)
        self.btn_rec.bind('<ButtonPress-1>', self.start_rec)
        self.btn_rec.bind('<ButtonRelease-1>', self.stop_rec)

        # Resultado
        self.lbl_result = tk.Label(self.root, text="---", 
                                   bg="#1e272e", fg="#0be881", 
                                   font=("Helvetica", 28, "bold"))
        self.lbl_result.pack(pady=20)
        
        self.lbl_conf = tk.Label(self.root, text="", bg="#1e272e", fg="#808e9b")
        self.lbl_conf.pack()

    def audio_callback(self, indata, frames, time, status):
        """Coleta o áudio em tempo real"""
        if self.recording:
            self.audio_buffer.append(indata.copy())

    def start_rec(self, event):
        if not self.model: return
        self.recording = True
        self.audio_buffer = []
        self.lbl_status.config(text="Ouvindo...", fg="#ffdd59")
        self.lbl_result.config(text="...")
        self.btn_rec.config(bg="#ff3f34") # Muda cor para indicar gravação
        
        # Inicia stream do sounddevice
        self.stream = sd.InputStream(callback=self.audio_callback, 
                                     channels=1, samplerate=utils.SR)
        self.stream.start()

    def stop_rec(self, event):
        if not self.recording: return
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.btn_rec.config(bg="#ff5e57")
        self.lbl_status.config(text="Processando...", fg="#4bcffa")
        
        # Processa em thread separada para não travar a janela
        threading.Thread(target=self.process_and_predict).start()

    def process_and_predict(self):
        if not self.audio_buffer: return

        # 1. Concatena os pedaços de áudio gravados
        audio_data = np.concatenate(self.audio_buffer, axis=0).flatten()

        # 2. Pré-processamento (Igual ao utils.py)
        # Normalização do áudio bruto
        max_val = np.max(np.abs(audio_data)) + 1e-9
        audio_data = audio_data / max_val

        # Gera Mel-Spectrograma
        S = librosa.feature.melspectrogram(
            y=audio_data, sr=utils.SR, 
            n_fft=utils.N_FFT, hop_length=utils.HOP_LENGTH, n_mels=utils.N_MELS
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # 3. Transforma em imagem (IMPORTADO DO UTILS)
        img = utils.spectrogram_to_image(S_db)

        # 4. Prepara para PyTorch (0-1, Transpose, Batch)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # (C, H, W)
        input_tensor = torch.tensor(img).unsqueeze(0).to(self.device)

        # 5. Inferência
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        # Atualiza a tela (tem que ser via root.after para thread safe)
        idx = predicted_idx.item()
        conf_val = confidence.item()
        
        classe_nome = utils.CLASSES[idx].upper()
        
        self.root.after(0, lambda: self.update_labels(classe_nome, conf_val))

    def update_labels(self, text, conf):
        self.lbl_result.config(text=text)
        self.lbl_conf.config(text=f"Confiança: {conf*100:.1f}%")
        self.lbl_status.config(text="Segure para falar", fg="#d2dae2")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()