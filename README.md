# 🎙️ Non-Verbal Audio Gestures Recognition

Sistema de reconhecimento de gestos não-verbais por áudio usando Deep Learning. O projeto identifica sons como assobios, estalos e palmas através de uma interface gráfica local e de um pipeline de treino voltado para batch.

## 🚀 Características

- **Classificação de Áudio em Tempo Real**: Reconhece gestos não-verbais através do microfone
- **Interface Gráfica**: Push-to-talk simples e responsiva usando Tkinter
- **CNN Personalizada**: Modelo de Convolutional Neural Network treinado em mel-spectrogramas
- **Data Augmentation**: Time-shifting, pitch-shifting e adição de ruído para melhorar generalização
- **Arquitetura Modular**: Código organizado e reutilizável
- **Treino em HPC**: Scripts prontos para Singularity + SLURM

## 🖥️ Execução no NPAD

O projeto já está preparado para rodar no supercomputador com Singularity.

- Guia completo: [README_NPAD.md](README_NPAD.md)
- Job batch: `train_job.sh`
- Container: `cnn_env.def`

## 📋 Pré-requisitos

- Python 3.8+
- Miniconda/Anaconda (opcional, mas recomendado)
- Microfone funcional
- PortAudio (para sounddevice)

### Instalação do PortAudio (Linux)

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev
```

## 🔧 Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/VictorAGomes/nonverbal-audio-gestures.git
cd nonverbal-audio-gestures
```

2. **Crie e ative o ambiente virtual**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📂 Estrutura do Projeto

```
nonverbal-audio-gestures/
├── config.py                 # Configurações globais centralizadas
├── model.py                  # Arquitetura CNN (NonVerbalCNN)
├── spectrogram_utils.py      # Conversão de espectrograma para imagem
├── train.py                  # Script de treinamento do modelo
├── gui_app.py                # Interface gráfica para inferência
├── main.py                   # Script de teste de mel-spectrograma
├── augmentation.py           # Funções de data augmentation
├── requirements.txt          # Dependências do projeto
├── best_<representation>.pth # Modelo treinado (após treinamento)
└── data/                     # Dados de treinamento
    ├── whistle/              # Áudios de assobio
    ├── snap/                 # Áudios de estalo
    └── clap/                 # Áudios de palmas
```

## 🎯 Como Usar

### 1. Preparar os Dados

Organize seus arquivos de áudio na estrutura:

```
data/
├── whistle/
│   ├── audio1.wav
│   ├── audio2.opus
│   └── ...
├── snap/
│   ├── audio1.wav
│   ├── audio2.mp3
│   └── ...
└── clap/
    └── ...
```

Formatos suportados: `.wav`, `.mp3`, `.opus`, `.flac`, `.m4a`, `.ogg`

Também há compatibilidade com nomes antigos em português: `assobio`, `dedo` e `palma`.

### 2. Treinar o Modelo

```bash
python train.py --data-dir data --output-dir outputs/local
```

O script irá:
- Carregar os dados da pasta `data/`
- Aplicar data augmentation
- Treinar a CNN por 50 épocas
- Salvar o melhor modelo como `best_<representation>.pth`
- Gerar matriz de confusão e curvas de aprendizado dentro de `outputs/local`

### 3. Executar a Interface Gráfica

```bash
python gui_app.py
```

**Como usar a interface:**
1. Pressione e **segure** o botão 🎙️
2. Faça o som (assobio ou estalo de dedos)
3. **Solte** o botão
4. Aguarde o resultado aparecer na tela

## 🧠 Arquitetura do Modelo

### NonVerbalCNN
```
Conv2D (3→32) → ReLU → MaxPool
Conv2D (32→64) → ReLU → MaxPool
Conv2D (64→128) → ReLU → MaxPool
Conv2D (128→256) → ReLU → MaxPool
Flatten
Dropout(0.3) → FC(16384→512) → ReLU
Dropout(0.5) → FC(512→num_classes)
```

### Pipeline de Processamento
1. **Áudio** (16kHz, mono, 1 segundo)
2. **Mel-Spectrograma** (128 bandas mel)
3. **Conversão para Imagem RGB** (128×128)
4. **Normalização** [0, 1]
5. **Inferência CNN**

## ⚙️ Configurações (config.py)

```python
SR = 16000              # Sample rate
DURATION = 1.0          # Duração em segundos
N_FFT = 1024            # Tamanho FFT
HOP_LENGTH = 256        # Passo entre janelas
N_MELS = 128            # Bandas Mel
IMG_SIZE = (128, 128)   # Tamanho da imagem
CLASSES = ['whistle', 'snap', 'clap']  # Classes reconhecidas internamente
```

## 📊 Data Augmentation

O treinamento aplica as seguintes técnicas:
- **Time Shifting** (70% chance): Desloca o sinal no tempo
- **Pitch Shifting** (50% chance): Altera a tonalidade
- **Background Noise** (80% chance): Adiciona ruído gaussiano

## 🔍 Testando Mel-Spectrograma

Para visualizar como o áudio é convertido:

```bash
python main.py
```

## 📈 Métricas de Avaliação

Após o treinamento, são gerados:
- **Acurácia de Teste**: Porcentagem de acertos no conjunto de teste
- **Matriz de Confusão**: Visualização de erros de classificação
- **Classification Report**: Precision, Recall e F1-Score por classe

## 🛠️ Desenvolvimento

### Adicionar Nova Classe

1. Crie uma pasta em `data/` com o nome da classe
2. Adicione arquivos de áudio nessa pasta
3. Atualize `config.py`:
```python
CLASSES = ['whistle', 'snap', 'clap', 'nova_classe']
```
4. Retreine o modelo: `python train.py`

### Estrutura Modular

- **`config.py`**: Centralize todas as constantes aqui
- **`model.py`**: Defina novas arquiteturas aqui
- **`spectrogram_utils.py`**: Funções de processamento de áudio
- **`train.py`**: Pipeline de treinamento


## 👥 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
1. Fazer fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abrir um Pull Request

## 📧 Contato

Dúvidas ou sugestões? Entre em contato através do GitHub.

---
