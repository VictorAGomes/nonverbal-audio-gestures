import os
import csv
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

from model import NonVerbalCNN
from spectrogram_utils import spectrogram_to_image
from config import SR, DURATION, N_FFT, HOP_LENGTH, N_MELS, IMG_SIZE, CLASSES, NUM_CLASSES, CLASS_TO_IDX

# =============================================================================
# REPRODUTIBILIDADE E HIPERPARÂMETROS
# =============================================================================
SEED                    = 42
EARLY_STOPPING_PATIENCE = 10    # epochs sem melhora antes de parar
EARLY_STOPPING_MIN_DELTA = 1e-4 # melhora mínima considerada significativa

np.random.seed(SEED)
torch.manual_seed(SEED)


class EarlyStopping:
    """Para o treino quando val_loss não melhora por `patience` epochs consecutivos."""
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# =============================================================================
# PRÉ-PROCESSAMENTO E AUGMENTATION
# =============================================================================
def load_and_normalize(path, sr=SR, duration=DURATION):
    """Carrega áudio completo, aplica pad simétrico ou crop centralizado."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    if duration is not None:
        target_length = int(sr * duration)
        if len(y) < target_length:
            pad_total = target_length - len(y)
            pad_left  = pad_total // 2
            pad_right = pad_total - pad_left
            y = np.pad(y, (pad_left, pad_right), 'constant')
        elif len(y) > target_length:
            start = (len(y) - target_length) // 2
            y = y[start:start + target_length]
    max_val = np.max(np.abs(y)) + 1e-9
    y = y / max_val
    return y, sr

def random_time_shift(y, shift_max=0.1):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def random_pitch_shift(y, sr, n_steps_range=(-2, 2)):
    n_steps = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_background_noise(y, noise_level_db_range=(-25, -15)):
    rms = np.sqrt(np.mean(y ** 2) + 1e-9)
    noise_db  = np.random.uniform(*noise_level_db_range)
    noise_rms = rms * 10 ** (noise_db / 20.0)
    noise     = np.random.normal(0, noise_rms, size=y.shape)
    y_noisy   = y + noise
    return y_noisy / (np.max(np.abs(y_noisy)) + 1e-9)

def augment_audio(y, sr):
    y = y.copy()
    if np.random.rand() < 0.7: y = random_time_shift(y)
    if np.random.rand() < 0.5: y = random_pitch_shift(y, sr)
    if np.random.rand() < 0.8: y = add_background_noise(y)
    return y

def extract_mel_spectrogram(y, sr):
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)

# =============================================================================
# DATASET PYTORCH
# =============================================================================
class AudioDataset(Dataset):
    def __init__(self, filepaths, labels, augment=False, representation='delta_delta'):
        self.filepaths      = filepaths
        self.labels         = labels
        self.augment        = augment
        self.representation = representation

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        y, sr = load_and_normalize(self.filepaths[idx])
        if self.augment:
            y = augment_audio(y, sr)
        S_db  = extract_mel_spectrogram(y, sr)
        image = spectrogram_to_image(S_db, representation=self.representation)
        image = np.transpose(image, (2, 0, 1))  # (H, W, 3) -> (3, H, W), já float32 [0,1]
        return torch.tensor(image), torch.tensor(self.labels[idx])

# =============================================================================
# CARREGAR DATASET
# =============================================================================
def load_dataset(data_dir):
    filepaths, labels = [], []
    AUDIO_EXTS = ('.wav', '.mp3', '.opus', '.flac', '.m4a', '.ogg')

    for class_name in CLASSES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Diretório não encontrado: {class_dir}")
            continue
        files = [f for ext in AUDIO_EXTS for f in glob.glob(os.path.join(class_dir, f"*{ext}"))]
        filepaths.extend(files)
        labels.extend([CLASS_TO_IDX[class_name]] * len(files))

    if len(filepaths) == 0:
        print("Nenhum arquivo nas pastas de classe. Tentando detectar por nome no diretório raiz...")
        for ext in AUDIO_EXTS:
            for path in glob.glob(os.path.join(data_dir, f"*{ext}")):
                name = os.path.basename(path).lower()
                for class_name in CLASSES:
                    if class_name in name:
                        filepaths.append(path)
                        labels.append(CLASS_TO_IDX[class_name])
                        break

    print(f"Total de arquivos carregados: {len(filepaths)}")
    return filepaths, labels

# =============================================================================
# TREINAMENTO — FOLD ÚNICO
# =============================================================================
def _train_single_fold(X_train, y_train, X_val, y_val,
                       epochs, batch_size, lr, representation, fold_id):
    """Treina um fold e retorna histórico de curvas e predições no val."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(
        AudioDataset(X_train, y_train, augment=True,  representation=representation),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        AudioDataset(X_val, y_val, augment=False, representation=representation),
        batch_size=batch_size, shuffle=False
    )

    model     = NonVerbalCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_acc   = 0.0
    checkpoint     = f'fold{fold_id}_{representation}.pth'
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    early_stopping = EarlyStopping()

    for epoch in range(epochs):
        # — Treino —
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for images, labels_b in tqdm(train_loader, desc=f'Fold {fold_id} Ep {epoch+1}/{epochs}', leave=False):
            images, labels_b = images.to(device), labels_b.to(device)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels_b)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            correct  += (out.argmax(1) == labels_b).sum().item()
            total    += labels_b.size(0)

        # — Validação —
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels_b in val_loader:
                images, labels_b = images.to(device), labels_b.to(device)
                out  = model(images)
                v_loss    += criterion(out, labels_b).item()
                v_correct += (out.argmax(1) == labels_b).sum().item()
                v_total   += labels_b.size(0)

        t_loss = run_loss / len(train_loader)
        v_loss = v_loss  / len(val_loader)
        t_acc  = 100 * correct   / total
        v_acc  = 100 * v_correct / v_total

        train_losses.append(t_loss);  val_losses.append(v_loss)
        train_accs.append(t_acc);     val_accs.append(v_acc)
        scheduler.step(v_loss)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), checkpoint)

        early_stopping.step(v_loss)
        print(f'  Fold {fold_id} | Epoch {epoch+1}/{epochs} | '
              f'Train {t_acc:.1f}% / Val {v_acc:.1f}% | '
              f'ES {early_stopping.counter}/{early_stopping.patience}')

        if early_stopping.should_stop:
            print(f'  Early stopping na epoch {epoch+1}.')
            break

    # — Predições no val com o melhor checkpoint —
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels_b in val_loader:
            images = images.to(device)
            preds  = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_b.numpy())

    os.remove(checkpoint)   # limpa checkpoint temporário do fold

    return {
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs':   train_accs,   'val_accs':   val_accs,
        'best_val_acc': best_val_acc,
        'preds':        all_preds,
        'labels':       all_labels,
    }

# =============================================================================
# TREINAMENTO — K-FOLD
# =============================================================================
def train_model_kfold(data_dir, epochs=50, batch_size=16, lr=0.0005,
                      representation='delta_delta', k=5):
    """K-fold stratified cross-validation com a NonVerbalCNN.

    Returns:
        dict com métricas agregadas, históricos de curvas e predições consolidadas.
    """
    filepaths, labels = load_dataset(data_dir)
    filepaths = np.array(filepaths)
    labels    = np.array(labels)

    skf           = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_histories = []
    all_preds, all_labels = [], []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), start=1):
        print(f'\n{"="*60}')
        print(f'  [{representation}] Fold {fold_id}/{k}')
        print(f'{"="*60}')

        history = _train_single_fold(
            filepaths[train_idx].tolist(), labels[train_idx].tolist(),
            filepaths[val_idx].tolist(),   labels[val_idx].tolist(),
            epochs=epochs, batch_size=batch_size, lr=lr,
            representation=representation, fold_id=fold_id,
        )
        fold_histories.append(history)
        all_preds.extend(history['preds'])
        all_labels.extend(history['labels'])

    # — Métricas agregadas —
    fold_accs = [h['best_val_acc'] for h in fold_histories]
    cv_acc_mean = float(np.mean(fold_accs))
    cv_acc_std  = float(np.std(fold_accs))

    # — Métricas por classe (sobre todas as predições agregadas) —
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    per_class = {cls: report[cls] for cls in CLASSES}

    # — Confusion matrix consolidada —
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix — {representation} ({k}-fold agregado)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{representation}.png', dpi=150)
    plt.close()

    print(f'\n[{representation}] CV Acc: {cv_acc_mean:.2f}% ± {cv_acc_std:.2f}%')
    print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))

    # — Salva o melhor fold como checkpoint final —
    best_fold_idx = int(np.argmax(fold_accs))
    _save_best_model(
        filepaths, labels, skf, best_fold_idx,
        epochs, batch_size, lr, representation,
    )

    return {
        'representation': representation,
        'cv_acc_mean':    cv_acc_mean,
        'cv_acc_std':     cv_acc_std,
        'fold_histories': fold_histories,
        'per_class':      per_class,
        'all_preds':      all_preds,
        'all_labels':     all_labels,
    }


def _save_best_model(filepaths, labels, skf, best_fold_idx,
                     epochs, batch_size, lr, representation):
    """Re-treina o melhor fold e salva como best_{representation}.pth."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = list(skf.split(filepaths, labels))
    train_idx, val_idx = splits[best_fold_idx]

    train_loader = DataLoader(
        AudioDataset(filepaths[train_idx].tolist(), labels[train_idx].tolist(),
                     augment=True, representation=representation),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        AudioDataset(filepaths[val_idx].tolist(), labels[val_idx].tolist(),
                     augment=False, representation=representation),
        batch_size=batch_size, shuffle=False
    )

    model     = NonVerbalCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    best_acc  = 0.0
    save_path = f'best_{representation}.pth'
    early_stopping = EarlyStopping()

    for epoch in range(epochs):
        model.train()
        for images, labels_b in train_loader:
            images, labels_b = images.to(device), labels_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels_b)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels_b in val_loader:
                images, labels_b = images.to(device), labels_b.to(device)
                out    = model(images)
                v_loss += criterion(out, labels_b).item()
                correct += (out.argmax(1) == labels_b).sum().item()
                total   += labels_b.size(0)

        v_loss_epoch = v_loss / len(val_loader)
        v_acc        = 100 * correct / total
        scheduler.step(v_loss_epoch)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_path)

        early_stopping.step(v_loss_epoch)
        if early_stopping.should_stop:
            print(f'  Early stopping na epoch {epoch+1}.')
            break

    print(f'Modelo final salvo: {save_path} (val acc: {best_acc:.2f}%)')

# =============================================================================
# CURVAS DE APRENDIZADO
# =============================================================================
def plot_learning_curves(fold_histories, representation):
    """Plota curvas de loss e acurácia (média ± desvio padrão entre folds).
    Folds com early stopping têm curvas mais curtas — usa o mínimo de epochs
    comuns a todos os folds para o cálculo da banda de desvio.
    """
    min_epochs = min(len(h['train_losses']) for h in fold_histories)
    x          = np.arange(1, min_epochs + 1)

    def mean_std(key):
        mat = np.array([h[key][:min_epochs] for h in fold_histories])
        return mat.mean(axis=0), mat.std(axis=0)

    t_loss_m, t_loss_s = mean_std('train_losses')
    v_loss_m, v_loss_s = mean_std('val_losses')
    t_acc_m,  t_acc_s  = mean_std('train_accs')
    v_acc_m,  v_acc_s  = mean_std('val_accs')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Learning Curves — {representation}', fontsize=13)

    # Loss
    ax = axes[0]
    ax.plot(x, t_loss_m, label='Train', color='steelblue')
    ax.fill_between(x, t_loss_m - t_loss_s, t_loss_m + t_loss_s, alpha=0.2, color='steelblue')
    ax.plot(x, v_loss_m, label='Val',   color='tomato')
    ax.fill_between(x, v_loss_m - v_loss_s, v_loss_m + v_loss_s, alpha=0.2, color='tomato')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(x, t_acc_m, label='Train', color='steelblue')
    ax.fill_between(x, t_acc_m - t_acc_s, t_acc_m + t_acc_s, alpha=0.2, color='steelblue')
    ax.plot(x, v_acc_m, label='Val',   color='tomato')
    ax.fill_between(x, v_acc_m - v_acc_s, v_acc_m + v_acc_s, alpha=0.2, color='tomato')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'learning_curves_{representation}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Curva de aprendizado salva: {path}')


def plot_ablation_comparison(results):
    """Gráfico de barras comparando as 3 representações (mean ± std)."""
    labels = [r['representation'] for r in results]
    means  = [r['cv_acc_mean']    for r in results]
    stds   = [r['cv_acc_std']     for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=['#4a90d9', '#e67e22', '#2ecc71'], alpha=0.85)
    ax.set_ylabel('CV Accuracy (%)')
    ax.set_title('Ablation Study — Representação do Espectrograma')
    ax.set_ylim(0, 105)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ablation_comparison.png', dpi=150)
    plt.close()
    print('Gráfico de comparação salvo: ablation_comparison.png')

# =============================================================================
# ABLATION STUDY
# =============================================================================
def run_ablation(data_dir, epochs=50, batch_size=16, k=5):
    """Treina a NonVerbalCNN com k-fold para cada representação e salva resultados."""
    representations = ['mel', 'delta', 'delta_delta']
    all_results = []

    for representation in representations:
        print(f'\n{"#"*60}')
        print(f'  REPRESENTAÇÃO: {representation}')
        print(f'{"#"*60}')
        result = train_model_kfold(
            data_dir, epochs=epochs, batch_size=batch_size,
            representation=representation, k=k,
        )
        plot_learning_curves(result['fold_histories'], representation)
        all_results.append(result)

    # — CSV detalhado com métricas por classe —
    csv_path = 'ablation_results.csv'
    fieldnames = ['representation', 'cv_acc_mean', 'cv_acc_std']
    for cls in CLASSES:
        fieldnames += [f'{cls}_precision', f'{cls}_recall', f'{cls}_f1']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                'representation': r['representation'],
                'cv_acc_mean':    round(r['cv_acc_mean'], 2),
                'cv_acc_std':     round(r['cv_acc_std'],  2),
            }
            for cls in CLASSES:
                m = r['per_class'][cls]
                row[f'{cls}_precision'] = round(m['precision'], 3)
                row[f'{cls}_recall']    = round(m['recall'],    3)
                row[f'{cls}_f1']        = round(m['f1-score'],  3)
            writer.writerow(row)

    # — Tabela resumo no terminal —
    print(f'\n{"="*70}')
    print('ABLATION STUDY — RESULTADOS FINAIS')
    print(f'{"="*70}')
    print(f"{'Repr.':<14} {'CV Acc':>12}", end='')
    for cls in CLASSES:
        print(f'  {cls[:6]:>6} F1', end='')
    print()
    print('-' * 70)
    for r in all_results:
        print(f"{r['representation']:<14} "
              f"{r['cv_acc_mean']:>6.2f}±{r['cv_acc_std']:.2f}%", end='')
        for cls in CLASSES:
            print(f"  {r['per_class'][cls]['f1-score']:>8.3f}", end='')
        print()
    print('=' * 70)
    print(f'\nResultados salvos em: {csv_path}')

    plot_ablation_comparison(all_results)
    return all_results

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    data_dir = "data"

    print("Iniciando Ablation Study — Non-Verbal Audio Gestures")
    print(f"Classes: {CLASSES}")
    print(f"Mel Spectrogram: {N_MELS} mel bands | Image size: {IMG_SIZE}")

    run_ablation(data_dir, epochs=50, batch_size=16, k=5)
