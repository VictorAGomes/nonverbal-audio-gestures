# Execução no NPAD com Singularity

Este projeto foi ajustado para rodar em batch no supercomputador usando `Singularity + SLURM`.

## O que mudou

- O treino agora aceita argumentos de linha de comando.
- Os resultados passam a ser salvos em um diretório de saída dedicado.
- O carregamento dos dados aceita tanto `whistle/snap/clap` quanto `assobio/dedo/palma`.
- O container foi simplificado para treino em ambiente headless.
- O job `train_job.sh` já está pronto para submissão no cluster.

## Estrutura esperada dos dados

O repositório atual usa esta estrutura:

```text
data/
├── whistle/
├── snap/
└── clap/
```

Também há compatibilidade com aliases em português:

- `whistle` -> `assobio`
- `snap` -> `dedo`
- `clap` -> `palma`

## 1. Gerar o container `.sif`

No NPAD, a criação de um container a partir de `.def` normalmente exige root e deve ser feita fora do cluster. Então gere a imagem na sua máquina e depois copie o `.sif` para o NPAD:

```bash
sudo singularity build cnn_env.sif cnn_env.def
```

O arquivo de definição usa:

- base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- dependências de treino de `requirements-train.txt`

## 2. Copiar projeto e container para o NPAD

Copie o repositório e o container:

```bash
scp -r nonverbal-audio-gestures usuario@npad:/caminho/do/projeto
scp cnn_env.sif usuario@npad:/caminho/do/projeto
```

No NPAD, deixe a imagem ao lado do projeto ou ajuste a variável `CONTAINER_PATH`.

## 3. Submeter o job

Entre no diretório do projeto e submeta:

```bash
sbatch train_job.sh
```

O script usa por padrão:

- `1` GPU
- `8` CPUs
- `32G` de RAM
- `24h` de tempo máximo
- saída em `outputs/<job_id>/`

## 4. Customizar sem editar o script

Você pode sobrescrever parâmetros no momento da submissão:

```bash
sbatch --export=ALL,\
CONTAINER_PATH=$PWD/cnn_env.sif,\
DATA_DIR=$PWD/data,\
EPOCHS=30,\
BATCH_SIZE=32,\
KFOLDS=5,\
NUM_WORKERS=8,\
REPRESENTATIONS="delta_delta" \
train_job.sh
```

## 5. Acompanhar a execução

Ver fila:

```bash
squeue -u $USER
```

Ver log:

```bash
tail -f slurm-<job_id>.out
```

## 6. Onde ficam os resultados

Ao final do job, os artefatos ficam em:

```text
outputs/<job_id>/
```

Arquivos esperados:

- `best_<representation>.pth`
- `confusion_matrix_<representation>.png`
- `learning_curves_<representation>.png`
- `ablation_results.csv`
- `ablation_comparison.png`
- `run_metadata.txt`

## 7. Comando equivalente para teste manual

Se quiser testar o container sem SLURM:

```bash
module load singularity
module load compilers/nvidia/cuda/12.6

singularity exec --nv \
  --bind "$PWD:$PWD" \
  --pwd "$PWD" \
  ./cnn_env.sif \
  python train.py \
  --data-dir data \
  --output-dir outputs/manual \
  --epochs 1 \
  --k-folds 2 \
  --representations delta_delta
```

## Observações

- A GUI (`gui_app.py`) não é o alvo do cluster.
- O treinamento salva gráficos sem depender de display gráfico.
- Se o dataset estiver desbalanceado demais, reduza `--k-folds` para não exceder a menor classe.
