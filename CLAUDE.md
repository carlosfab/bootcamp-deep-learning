# CLAUDE.md — Lung Cancer Detection (LUNA16)

> Documento de contexto e planejamento para agentes de IA que auxiliam neste projeto.

## Regras de Estilo

Todos os agentes que trabalharem neste projeto devem seguir estas regras:

- **Nunca usar emojis** em qualquer output (codigo, markdown, notebooks, comentarios).
- **Notebooks**: nao usar secoes numeradas (ex: "1. Introducao", "2. Dados"). Usar apenas titulos descritivos.
- **Notebooks**: nao usar separadores horizontais (`---`) entre secoes.
- Manter linguagem tecnica, direta e sem decoracoes visuais desnecessarias.

## Objetivo do Projeto

Desenvolver um pipeline completo de **deteccao de cancer de pulmao** usando Deep Learning, desde o pre-processamento de CT scans ate a classificacao de nodulos, utilizando o dataset LUNA16.

## Estrutura

| Diretorio    | Proposito                                      |
|-------------|------------------------------------------------|
| `data/luna/` | Dataset LUNA16 processado                      |
| `data/raw/`  | Dados brutos / CT scans originais              |
| `notebooks/` | Exploracao, experimentacao e visualizacao       |
| `src/`       | Codigo-fonte modularizado do pipeline          |
| `tests/`     | Testes unitarios e de integracao               |
| `docs/`      | Documentacao tecnica e referencias             |

## Stack Tecnica

- **Python** >= 3.11.3
- **PyTorch** (torch, torchvision, torchaudio)
- **SimpleITK** — leitura e processamento de imagens medicas (`.mhd` / `.raw`)
- **scikit-image / scikit-learn** — pre-processamento e metricas
- **Matplotlib / Plotly** — visualizacao
- **Gradio** — interface de demonstracao
- **UV** — gerenciamento de ambiente virtual
- **Pytest** — testes

## Backlog / Demandas

### Concluido
- [x] Estrutura inicial do projeto criada
- [x] Ambiente virtual configurado com UV
- [x] Repositorio Git inicializado

### Em Andamento
- [ ] Planejamento detalhado do pipeline

### Pendente
- [ ] Download e organizacao do dataset LUNA16
- [ ] Exploracao e visualizacao dos CT scans
- [ ] Pre-processamento dos dados (resampling, normalizacao, segmentacao)
- [ ] Geracao de candidatos a nodulos
- [ ] Treinamento do modelo de classificacao
- [ ] Avaliacao e metricas (FROC, sensibilidade)
- [ ] Interface Gradio para demonstracao

## Notas e Decisoes

_Secao para registrar decisoes de design, trade-offs e aprendizados ao longo do desenvolvimento._

*Ultima atualizacao: 2026-03-01*
