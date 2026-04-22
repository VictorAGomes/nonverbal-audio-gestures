# Execução no NPAD com Singularity

Este projeto foi simplificado para rodar no NPAD usando `Singularity + SLURM`, sem argumentos de linha de comando.

Hoje o fluxo real é este:

- o `train.py` usa `data/` como diretório de entrada;
- os parâmetros principais já ficam definidos no próprio código;
- o job `train_job.sh` apenas executa `python train.py` dentro do container;
- os arquivos gerados são salvos no diretório do projeto.

## O que o `train.py` faz hoje

Na execução atual, o bloco principal do script:

- usa `data_dir = "data"`;
- executa `run_ablation_arch(...)`;
- roda o estudo arquitetural com `epochs=50`;
- usa `batch_size=16`;
- usa `k=5`.

Ou seja: não usamos `--data-dir`, `--epochs`, `--output-dir` ou outros parâmetros via terminal.

Se você quiser mudar dataset, número de épocas, batch size ou quantidade de folds, faça isso diretamente no `train.py`.

## Estrutura esperada dos dados

O formato principal esperado é:

```text
data/
├── whistle/
├── snap/
└── clap/
```

O carregamento considera estas classes:

- `whistle`
- `snap`
- `clap`

Extensões aceitas:

- `.wav`
- `.mp3`
- `.opus`
- `.flac`
- `.m4a`
- `.ogg`

Se essas pastas não existirem, o script ainda tenta detectar a classe pelo nome do arquivo dentro da raiz de `data/`, desde que o nome contenha `whistle`, `snap` ou `clap`.

## 1. Gerar o container `.sif`

Em geral, a imagem é gerada fora do NPAD e depois copiada para o cluster:

```bash
sudo singularity build cnn_env.sif cnn_env.def
```

O container usa:

- base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- dependências de `requirements-train.txt`

## 2. Copiar projeto e container para o NPAD

```bash
scp -r nonverbal-audio-gestures usuario@npad:/caminho/do/projeto
scp cnn_env.sif usuario@npad:/caminho/do/projeto
```

Deixe o `cnn_env.sif` no mesmo diretório do projeto no cluster, ou ajuste `CONTAINER_PATH` no momento da submissão.

## 3. Submeter o job

No NPAD, entre no diretório do projeto e execute:

```bash
sbatch train_job.sh
```

O `train_job.sh`:

- carrega os módulos do ambiente;
- monta o diretório do projeto no container;
- executa `python train.py`.

Recursos configurados no job:

- `1` GPU
- `8` CPUs
- `32G` de RAM
- `24h` de tempo máximo

## 4. Execução manual sem SLURM

Se quiser testar manualmente no NPAD:

```bash
module load singularity
module load compilers/nvidia/cuda/12.6

singularity exec --nv \
  --bind "$PWD:$PWD" \
  --pwd "$PWD" \
  ./cnn_env.sif \
  python train.py
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

## 6. Onde os resultados ficam

Os arquivos são salvos no diretório atual do projeto, não em `outputs/<job_id>/`.

Arquivos gerados pelo fluxo atual incluem:

- `ablation_arch_results.csv`
- `ablation_comparison.png`
- `learning_curves_<nome_da_arquitetura>.png`
- `confusion_matrix_mel.png`
- `best_mel.pth`

Como o script roda várias arquiteturas em sequência, alguns arquivos com nome fixo podem ser sobrescritos ao longo da execução.

## 7. Como alterar o experimento

As mudanças são feitas diretamente no `train.py`.

Exemplos comuns:

- trocar `data_dir = "data"` por outro caminho;
- ajustar `epochs`, `batch_size` e `k`;
- trocar `run_ablation_arch(...)` por `run_ablation(...)` se quiser comparar representações em vez de arquiteturas.

## Observações

- A execução no cluster é voltada para treino, não para `gui_app.py`.
- O salvamento de gráficos funciona em ambiente headless.
- Para `k` folds, cada classe precisa ter amostras suficientes para a divisão estratificada funcionar corretamente.
