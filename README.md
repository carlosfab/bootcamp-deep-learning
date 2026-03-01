# 🫁 Detecção de Câncer de Pulmão — LUNA16

> Projeto de Deep Learning para detecção de nódulos pulmonares utilizando o dataset **LUNA16** (Lung Nodule Analysis 2016).

## Estrutura do Projeto

```
bootcamp-deep-learning/
├── data/
│   ├── luna/          # Dataset LUNA16 processado
│   └── raw/           # Dados brutos (CT scans)
├── notebooks/         # Jupyter notebooks (exploração e experimentos)
├── src/               # Código-fonte do projeto
├── tests/             # Testes automatizados
├── docs/              # Documentação do projeto
├── .env               # Variáveis de ambiente (não versionado)
├── pyproject.toml     # Dependências e configuração do projeto
└── CLAUDE.md          # Contexto e planejamento para agentes de IA
```

## Setup

```bash
# Clonar o repositório
git clone https://github.com/carlosfab/bootcamp-deep-learning.git
cd bootcamp-deep-learning

# Criar ambiente virtual e instalar dependências
uv sync

# Ativar o ambiente
source .venv/bin/activate
```

## Dataset

O projeto utiliza o [LUNA16](https://luna16.grand-challenge.org/) — um benchmark público para detecção de nódulos pulmonares em tomografias computadorizadas (CT scans).

## Licença

Este projeto é parte do Bootcamp de Deep Learning para Visão Computacional da [Sigmoidal](https://sigmoidal.ai).
