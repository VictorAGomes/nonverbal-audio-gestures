import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# CONFIGURAÇÕES GLOBAIS

SR = 16000              # Sample rate
DURATION = 1.0          # Duração em segundos
N_FFT = 1024            # Tamanho FFT
HOP_LENGTH = 256        # Passo entre janelas
N_MELS = 128            # Bandas Mel
IMG_SIZE = (128, 128)   # Tamanho final da imagem

CLASSES = ['assobio', 'dedo']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# 2. NORMALIZAÇÃO E AUGMENTATION 
def load_and_normalize(path, sr=SR, duration=DURATION):
    y, sr = librosa.load(path, sr=sr, duration=duration, mono=True)
    # Pad ou trim para duração fixa
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    else:
        y = y[:target_length]
    # Normalização
    max_val = np.max(np.abs(y)) + 1e-9
    y = y / max_val
    return y, sr

def random_time_shift(y, shift_max=0.1):
    n = len(y)
    shift = int(np.random.uniform(-shift_max, shift_max) * n)
    return np.roll(y, shift)

def random_pitch_shift(y, sr, n_steps_range=(-2, 2)):
    n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_background_noise(y, noise_level_db_range=(-25, -15)):
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    noise_db = np.random.uniform(noise_level_db_range[0], noise_level_db_range[1])
    noise_rms = rms * 10**(noise_db / 20.0)
    noise = np.random.normal(0, noise_rms, size=y.shape)
    y_noisy = y + noise
    max_val = np.max(np.abs(y_noisy)) + 1e-9
    return y_noisy / max_val

def augment_audio(y, sr):
    y_aug = y.copy()
    if np.random.rand() < 0.7:
        y_aug = random_time_shift(y_aug)
    if np.random.rand() < 0.5:
        y_aug = random_pitch_shift(y_aug, sr)
    if np.random.rand() < 0.8:
        y_aug = add_background_noise(y_aug)
    return y_aug

# 3. EXTRACAO DE MEL-SPECTROGRAMA
def extract_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def spectrogram_to_image(S_db, target_size=IMG_SIZE):
    """Renderiza o espectrograma para uma imagem RGB de tamanho target_size.
    Compatível com diferentes backends matplotlib no macOS."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from PIL import Image

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

# DATASET PYTORCH
class AudioDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None, augment=False):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        
        # Carrega e normaliza
        y, sr = load_and_normalize(path)
        
        # Augmentation (apenas treino)
        if self.augment:
            y = augment_audio(y, sr)
        
        # Extrai mel-spectrograma
        S_db = extract_mel_spectrogram(y, sr)
        
        # Converte para imagem
        image = spectrogram_to_image(S_db)
        
        # Normaliza para [0,1]
        image = image.astype(np.float32) / 255.0
        
        # Para PyTorch: C x H x W
        image = np.transpose(image, (2, 0, 1))
        
        return torch.tensor(image), torch.tensor(label)

# MODELO CNN
class NonVerbalCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(NonVerbalCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv1: 3->32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv2: 32->64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv3: 64->128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv4: 128->256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# CARREGAR DATASET
def load_dataset(data_dir):
    """Carrega arquivos de todas as classes (procura por extensões comuns)."""
    filepaths, labels = [], []
    AUDIO_EXTS = ('.wav', '.mp3', '.opus', '.flac', '.m4a', '.ogg')
    
    for class_name in CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Diretório não encontrado: {class_dir}")
            continue
        
        class_files = []
        for ext in AUDIO_EXTS:
            class_files.extend(glob.glob(os.path.join(class_dir, f"*{ext}")))
        
        class_idx = CLASS_TO_IDX[class_name]
        filepaths.extend(class_files)
        labels.extend([class_idx] * len(class_files))
    
    # Se nada foi encontrado nas pastas de classe, tenta detectar arquivos na raiz usando o nome do arquivo
    if len(filepaths) == 0:
        print("Nenhum arquivo nas pastas de classe. Tentando detectar arquivos no diretório raiz por extensão/nome...")
        for ext in AUDIO_EXTS:
            for path in glob.glob(os.path.join(data_dir, f"*{ext}")):
                name = os.path.basename(path).lower()
                matched = False
                for class_name in CLASSES:
                    if class_name in name:
                        filepaths.append(path)
                        labels.append(CLASS_TO_IDX[class_name])
                        matched = True
                        break
                if not matched:
                    pass
    
    print(f"Total de arquivos carregados: {len(filepaths)}")
    return filepaths, labels

# TREINAMENTO
def train_model(data_dir, epochs=50, batch_size=32, lr=0.001):
    # Carrega dados
    filepaths, labels = load_dataset(data_dir)
    
    # Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Treino: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Datasets e DataLoaders
    train_dataset = AudioDataset(X_train, y_train, augment=True)
    val_dataset = AudioDataset(X_val, y_val, augment=False)
    test_dataset = AudioDataset(X_test, y_test, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Modelo, loss, otimizador
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NonVerbalCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Treinamento
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validação
        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Métricas
        train_loss = running_loss / len(train_loader)
        val_loss_epoch = val_loss / len(val_loader)
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        train_losses.append(train_loss)
        val_losses.append(val_loss_epoch)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss_epoch)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Salva melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_nonverbal_model.pth')
            print(f'Novo melhor modelo salvo! Val Acc: {val_acc:.2f}%')
    
    # Teste final
    model.load_state_dict(torch.load('best_nonverbal_model.pth'))
    model.eval()
    correct_test, total_test = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * correct_test / total_test
    print(f'\nTest Accuracy: {test_acc:.2f}%')
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Relatório de classificação
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    return model, {
        'train_acc': train_accs, 'val_acc': val_accs,
        'train_loss': train_losses, 'val_loss': val_losses,
        'test_acc': test_acc
    }

# =============================================================================
# 8. EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # CRIE A ESTRUTURA:
    # data/
    # ├── whistle/*.wav
    # ├── click/*.wav  
    # └── blow/*.wav
    
    data_dir = "data"  # Ajuste o caminho
    
    print("🚀 Iniciando treinamento Non-Verbal Audio CNN...")
    print(f"Classes: {CLASSES}")
    print(f"Mel Spectrogram: {N_MELS} mel bands")
    print(f"Image size: {IMG_SIZE}")
    
    model, history = train_model(data_dir, epochs=50, batch_size=32)
    
    print("\n✅ Treinamento concluído!")
    print("Modelo salvo como: best_nonverbal_model.pth")
    print("Confusion Matrix salva como: confusion_matrix.png")
