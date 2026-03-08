# Detecção de Câncer de Pulmão — LUNA16

<a href="https://starresearch.institute" alt="star"><img src="https://img.shields.io/badge/Bootcamp-Deep%20Learning-0D1117?style=flat&logo=pytorch&logoColor=EE4C2C" /></a>
<a href="http://linkedin.com/in/carlos-melo-data-science/" alt="linkedin"> <img src="https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white" /></a>
<a href="https://starresearch.institute" alt="star"> <img src="https://img.shields.io/badge/STAR%20Research%20Institute-1a2332" /></a>

Pipeline de classificação binária (nódulo vs não-nódulo) em tomografias computadorizadas, utilizando o dataset [LUNA16](https://luna16.grand-challenge.org/) e PyTorch.

![Banner](docs/fixed_cnn_lung_tumor_detection.png)

<br>

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Progresso](#progresso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação e Configuração](#instalação-e-configuração)

<br>

## Sobre o Projeto

O projeto implementa um pipeline completo de detecção de nódulos pulmonares a partir de tomografias computadorizadas (CT scans), desde a exploração e preparação dos dados até o deploy de uma aplicação interativa com Gradio.

```
Candidatos pré-computados (CSV) → Carregar CT → Extrair patch 3D → Classificar (CNN 3D) → Deploy Gradio
```

A abordagem utiliza **candidatos pré-computados** fornecidos pelo challenge LUNA16 (~551 mil coordenadas XYZ). Cada candidato é extraído como um crop 3D de 32×48×48 voxels e classificado por uma CNN 3D como nódulo ou não-nódulo.

Não fazemos segmentação nem detecção no pipeline principal — os candidatos já vêm pré-computados pelo challenge.

<br>

## Progresso

### Concluído

- [x] Download e organização do dataset LUNA16
- [x] Análise exploratória e unificação das fontes de dados
- [x] Carregamento de CT scans e conversão de coordenadas
- [x] Construção do PyTorch Dataset com extração de crops 3D
- [x] Arquitetura da CNN 3D para classificação de nódulos
- [x] Loop de treinamento com balanceamento e data augmentation

### Próximas etapas

- [ ] Treinamento completo em GPU
- [ ] Avaliação do modelo e análise de erros
- [ ] Deploy com Gradio

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
│   └── 06_training
├── src/                       Módulos Python (gerados via %%writefile)
│   ├── luna_data.py
│   ├── model.py
│   └── training.py
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
