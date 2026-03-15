# Detecção de Câncer de Pulmão

<a href="https://starresearch.institute" alt="star"><img src="https://img.shields.io/badge/Bootcamp-Deep%20Learning-0D1117?style=flat&logo=pytorch&logoColor=EE4C2C" /></a>
<a href="http://linkedin.com/in/carlos-melo-data-science/" alt="linkedin"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white" /></a>
<a href="https://starresearch.institute" alt="star"> <img src="https://img.shields.io/badge/STAR%20Research%20Institute-1a2332" /></a>

Pipeline de classificação binária (nódulo vs não-nódulo) em tomografias computadorizadas, utilizando o dataset [LUNA16](https://luna16.grand-challenge.org/) e PyTorch.

![Banner](docs/fixed_cnn_lung_tumor_detection.png)

<br>

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Tomografias Computadorizadas](#tomografias-computadorizadas)
- [Pipeline de Dados](#pipeline-de-dados)
- [Progresso](#progresso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação e Configuração](#instalação-e-configuração)

<br>

## Sobre o Projeto

O projeto implementa um pipeline completo de detecção de nódulos pulmonares a partir de tomografias computadorizadas (CT scans), desde a exploração e preparação dos dados até o deploy de uma aplicação interativa com Gradio.

<p align="center">
  <img src="docs/fixed_landing_7_technical_flow_light.png" alt="Visão geral do pipeline" width="85%">
</p>
<p align="center"><em>Visão geral do pipeline — do CT scan bruto até a classificação por uma CNN 3D.</em></p>

A abordagem utiliza **candidatos pré-computados** fornecidos pelo challenge LUNA16 (~551 mil coordenadas XYZ). Cada candidato é extraído como um crop 3D de 32x48x48 voxels e classificado por uma CNN 3D como nódulo ou não-nódulo. Não fazemos segmentação nem detecção no pipeline principal — os candidatos já vêm pré-computados pelo challenge.

<br>

## Tomografias Computadorizadas

<p align="center">
  <img src="docs/fixed_ct_slices_concept.png" alt="Slices de uma tomografia computadorizada" width="85%">
</p>
<p align="center"><em>Uma tomografia é composta por centenas de slices axiais empilhados, formando um volume 3D.</em></p>

Uma tomografia computadorizada (CT scan) gera um volume 3D do corpo do paciente. Cada "fatia" (slice) é uma imagem 2D, e a pilha de fatias forma o volume completo. Os valores de cada voxel são medidos em **Unidades Hounsfield (HU)** — uma escala onde o ar vale -1000 HU, a água vale 0 HU e o osso pode chegar a +1000 HU.

No dataset LUNA16, cada CT scan é armazenado como um par de arquivos `.mhd` (metadados) e `.raw` (voxels). O desafio fornece dois CSVs: `candidates.csv` com ~551 mil coordenadas XYZ de pontos suspeitos, e `annotations.csv` com os nódulos confirmados por radiologistas.

<br>

## Pipeline de Dados

<p align="center">
  <img src="docs/fixed_lung_cancer_pipeline_oreilly.png" alt="Pipeline de dados" width="85%">
</p>
<p align="center"><em>Pipeline completo: dos arquivos brutos até o sample pronto para a rede neural.</em></p>

O caminho dos dados brutos até a entrada da rede neural segue estas etapas:

1. **Carregar o CT scan** — leitura do `.mhd` com SimpleITK, obtendo o array 3D e os metadados (origin, spacing, direction)
2. **Converter coordenadas** — as coordenadas XYZ (milímetros do paciente) são convertidas para índices IRC (index, row, col) do array NumPy
3. **Extrair o crop 3D** — um patch de 32x48x48 voxels é recortado ao redor de cada candidato
4. **Criar o sample PyTorch** — o crop vira um tensor `[1, 32, 48, 48]`, pronto para o DataLoader

<br>

## Progresso

- [x] Download e organização do dataset LUNA16
- [x] Análise exploratória e unificação das fontes de dados
- [x] Carregamento de CT scans e conversão de coordenadas
- [x] Construção do PyTorch Dataset com extração de crops 3D
- [x] Arquitetura da CNN 3D para classificação de nódulos
- [x] Loop de treinamento com balanceamento e data augmentation
- [x] Treinamento completo em GPU
- [x] Avaliação do modelo e análise de erros
- [x] Deploy com Gradio

<br>

## Estrutura do Projeto

```
.
├── notebooks/                 Jupyter notebooks do curso
│   ├── 01_download_luna16
│   ├── 02_explore_csv_data
│   ├── 03_analyze_coordinates
│   ├── 04_ct_scan_to_dataset
│   ├── 05_model_architecture
│   ├── 06_training
│   ├── 07_colab_training
│   ├── 08_model_evaluation
│   └── 09_gradio_deploy
├── src/                       Módulos Python (gerados via %%writefile)
│   ├── luna_data.py
│   ├── model.py
│   ├── training.py
│   └── inference.py
├── app.py                     Aplicação Gradio (gerado via %%writefile)
├── checkpoints/               Checkpoints do modelo (não versionado)
├── data/                      Dataset LUNA16 (não versionado)
├── docs/                      Diagramas e referências
└── pyproject.toml             Dependências e configuração
```

<br>

## Instalação e Configuração

1. Clonar o repositório para a sua máquina local:

```bash
git clone https://github.com/carlosfab/bootcamp-deep-learning.git
cd bootcamp-deep-learning
```

2. Instalar as dependências com [UV](https://docs.astral.sh/uv/):

```bash
uv sync
```

3. Ativar o ambiente virtual:

```bash
source .venv/bin/activate
```

4. O dataset LUNA16 (~111 GB) deve ser baixado separadamente. O notebook `01_download_luna16.ipynb` contém as instruções de download via API.

---

Projeto desenvolvido como parte do Bootcamp de Deep Learning para Visão Computacional da [STAR Research Institute](https://starresearch.institute).
