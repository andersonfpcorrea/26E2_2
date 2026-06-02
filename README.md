# Pipeline Completo de PLN — Corpus BBC News

Projeto da disciplina **Processamento de Linguagem Natural** (Sistemas
Cognitivos e Linguagem Natural).

Pipeline de PLN de ponta a ponta sobre o corpus **BBC News** (2.225 notícias, 5 editorias): pré-processamento,
representação vetorial, busca textual, classificação, modelagem de tópicos,
extração de entidades e grafo de conhecimento.

## Artefatos entregues

| Artefato                         | Caminho                                                                |
| -------------------------------- | ---------------------------------------------------------------------- |
| Notebook principal (executado)   | `notebooks/pln_pipeline.ipynb`                                         |
| Notebook em HTML estático        | `notebooks/pln_pipeline.html`                                          |
| Relatório técnico (PDF)          | `report/anderson_correa_sistemas-cognitivos-linguagem-natural_pln.pdf` |
| Corpus consolidado               | `data/bbc_news.csv`                                                    |
| Dados pré-processados (contrato) | `data/processed.parquet`                                               |
| Figuras exportadas               | `outputs/figures/`                                                     |
| Modelos salvos                   | `outputs/models/`                                                      |
| Grafo de conhecimento            | `outputs/graph/` (`.graphml` + `.html` interativo)                     |

O notebook está versionado **com resultados calculados**.

## Estrutura

```
.
├── notebooks/pln_pipeline.ipynb     # pipeline
├── report/                          # relatório (Markdown + PDF)
├── data/                            # corpus + dados pré-processados
├── outputs/{figures,models,graph}/  # artefatos gerados
├── scripts/                         # download, pré-processamento, setup, PDF
├── build/                           # módulos do notebook (scripts + narrativas)
└── pyproject.toml / uv.lock         # ambiente uv (Python 3.12)
```

## Reprodução

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12 (via uv).

```bash
# 1. Ambiente + dependências (a partir do uv.lock)
uv sync

# 2. Dados do NLTK -> nltk_data/ (~117 MB, não versionado)
uv run python scripts/setup_nltk.py

# 3. (Opcional) Reconstruir o corpus a partir da fonte -> data/bbc_news.csv
uv run python scripts/download_corpus.py

# 4. (Opcional) Reconstruir o contrato de pré-processamento -> data/processed.parquet
uv run python scripts/build_processed.py

# 5. Executar o notebook (regenera figuras, modelos e grafo)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/pln_pipeline.ipynb

# 6. Exportar a versão estática em HTML -> notebooks/pln_pipeline.html
uv run jupyter nbconvert --to html notebooks/pln_pipeline.ipynb

# 7. Gerar o relatório em PDF
uv run --extra report python scripts/build_report.py
```

Para abrir o notebook no navegador

```bash
uv run jupyter lab notebooks/pln_pipeline.ipynb
```

Como alternativa sem servidor Jupyter, basta abrir `notebooks/pln_pipeline.html`
diretamente no navegador.

As etapas 3 e 4 são opcionais porque `data/bbc_news.csv` e
`data/processed.parquet` já estão versionados. As etapas 2, 5 e 6 só são
necessárias para reexecutar o notebook do zero.

## Corpus

BBC News — D. Greene e P. Cunningham, _"Practical Solutions to the Problem of
Diagonal Dominance in Kernel Document Clustering"_, ICML 2006.
Fonte: http://mlg.ucd.ie/datasets/bbc.html
