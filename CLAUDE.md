# CLAUDE.md — Lung Cancer Detection (LUNA16)

> Documento de contexto e planejamento para agentes de IA que auxiliam neste projeto.

## 🎯 Objetivo do Projeto

Desenvolver um pipeline completo de **detecção de câncer de pulmão** usando Deep Learning, desde o pré-processamento de CT scans até a classificação de nódulos, utilizando o dataset LUNA16.

## 📁 Estrutura

| Diretório    | Propósito                                      |
|-------------|------------------------------------------------|
| `data/luna/` | Dataset LUNA16 processado                      |
| `data/raw/`  | Dados brutos / CT scans originais              |
| `notebooks/` | Exploração, experimentação e visualização      |
| `src/`       | Código-fonte modularizado do pipeline          |
| `tests/`     | Testes unitários e de integração               |
| `docs/`      | Documentação técnica e referências             |

## 🔧 Stack Técnica

- **Python** ≥ 3.11.3
- **PyTorch** (torch, torchvision, torchaudio)
- **SimpleITK** — leitura e processamento de imagens médicas (`.mhd` / `.raw`)
- **scikit-image / scikit-learn** — pré-processamento e métricas
- **Matplotlib / Plotly** — visualização
- **Gradio** — interface de demonstração
- **UV** — gerenciamento de ambiente virtual
- **Pytest** — testes

## 📋 Backlog / Demandas

### 🟢 Concluído
- [x] Estrutura inicial do projeto criada
- [x] Ambiente virtual configurado com UV
- [x] Repositório Git inicializado

### 🟡 Em Andamento
- [ ] Planejamento detalhado do pipeline

### 🔴 Pendente
- [ ] Download e organização do dataset LUNA16
- [ ] Exploração e visualização dos CT scans
- [ ] Pré-processamento dos dados (resampling, normalização, segmentação)
- [ ] Geração de candidatos a nódulos
- [ ] Treinamento do modelo de classificação
- [ ] Avaliação e métricas (FROC, sensibilidade)
- [ ] Interface Gradio para demonstração

## 📝 Notas e Decisões

_Seção para registrar decisões de design, trade-offs e aprendizados ao longo do desenvolvimento._

---

*Última atualização: 2026-03-01*
