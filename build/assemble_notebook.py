"""Assemble the end-to-end notebook from verified section scripts + narratives.

Each ``build/sectionN_*.py`` is split on its ``# === CELL:`` markers into code
cells; Portuguese narrative markdown is interleaved; figure-display cells embed
the saved PNGs inline. Two notebook-incompatible constructs are rewritten:
``Path(__file__)`` (no ``__file__`` in a kernel) and the ``if __name__`` guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
NARR = BUILD / "narratives"


def read(path: Path) -> str:
    """Return the text content of a file."""
    return path.read_text(encoding="utf-8")


def strip_docstring(source: str) -> str:
    """Remove the leading module docstring from a Python source string."""
    stripped = source.lstrip()
    if stripped.startswith('"""'):
        first = source.index('"""')
        second = source.index('"""', first + 3)
        source = source[second + 3:]
    return source.lstrip("\n")


def transform(code: str) -> str:
    """Rewrite constructs that do not work inside a Jupyter kernel."""
    code = code.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
        "PROJECT_ROOT = NB_ROOT",
    )
    code = re.sub(
        r'if __name__ == "__main__":\n\s+main\(\)\s*$',
        "main()\n",
        code,
    )
    return code.strip("\n")


def split_cells(path: Path) -> list[tuple[str, str]]:
    """Split a section script into (title, code) chunks by ``# === CELL:`` markers.

    Returns a list of (title, code) tuples. Any preamble before the first marker
    is returned with the title ``"Imports"``.
    """
    source = strip_docstring(read(path))
    parts = re.split(r"(?m)^# === CELL: (.*?) ===$\n", source)
    cells: list[tuple[str, str]] = []
    preamble = parts[0].strip("\n")
    if preamble.strip():
        cells.append(("Imports", transform(preamble)))
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = transform(parts[i + 1])
        if body.strip():
            cells.append((title, body))
    return cells


def code_cells_from(path: Path) -> list:
    """Build notebook code cells from a section script."""
    out = []
    for title, body in split_cells(path):
        out.append(new_code_cell(f"# --- {title} ---\n{body}"))
    return out


def figure_cell(filenames: list[str]) -> object:
    """Build a code cell that displays a list of saved figures inline."""
    listing = ", ".join(f'"{name}"' for name in filenames)
    src = (
        "# Display saved figures inline\n"
        f"for _name in [{listing}]:\n"
        "    display(Image(filename=str(FIG_DIR / _name)))"
    )
    return new_code_cell(src)


MASTER_SETUP = '''\
# --- Master setup: imports, paths, reproducibility ---
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import nltk
import numpy as np
import pandas as pd
from IPython.display import Image, display

# Resolve the project root whether the kernel starts in the repo root or notebooks/.
NB_ROOT = Path.cwd()
if not (NB_ROOT / "data" / "processed.parquet").exists():
    NB_ROOT = NB_ROOT.parent
assert (NB_ROOT / "data" / "processed.parquet").exists(), (
    "processed.parquet not found - run scripts/build_processed.py first"
)

nltk.data.path.insert(0, str(NB_ROOT / "nltk_data"))

PROJECT_ROOT = NB_ROOT
DATA_DIR = NB_ROOT / "data"
FIG_DIR = NB_ROOT / "outputs" / "figures"
MODEL_DIR = NB_ROOT / "outputs" / "models"
GRAPH_DIR = NB_ROOT / "outputs" / "graph"
for _d in (FIG_DIR, MODEL_DIR, GRAPH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DF = pd.read_parquet(DATA_DIR / "processed.parquet")
print(f"Corpus loaded: {DF.shape[0]} documents, {DF['category'].nunique()} categories")

import sklearn
import spacy
import gensim
import networkx
print(f"pandas {pd.__version__} | numpy {np.__version__} | scikit-learn {sklearn.__version__}")
print(f"spaCy {spacy.__version__} | gensim {gensim.__version__} | networkx {networkx.__version__}")
'''

GRAPH_HTML_NOTE = (
    "O grafo também foi exportado em formato **interativo** "
    "(`outputs/graph/knowledge_graph.html`, abrível no navegador) e em "
    "**GraphML** (`outputs/graph/knowledge_graph.graphml`) para análise externa."
)


def build() -> None:
    """Assemble and write the end-to-end notebook."""
    cells: list = []

    cells.append(new_markdown_cell(read(NARR / "00_header.md")))
    cells.append(new_markdown_cell(read(NARR / "01_setup.md")))
    cells.append(new_code_cell(MASTER_SETUP))

    # Sections 2-3: inspection + preprocessing live in one script.
    s23 = split_cells(ROOT / "build" / "section3_preprocessing.py")
    cells.append(new_markdown_cell(read(NARR / "02_inspection.md")))
    # All definition chunks (everything except the Main chunk).
    def_chunks = [(t, b) for t, b in s23 if t != "Main"]
    main_chunk = [(t, b) for t, b in s23 if t == "Main"]
    for title, body in def_chunks:
        cells.append(new_code_cell(f"# --- {title} ---\n{body}"))
    cells.append(new_markdown_cell(read(NARR / "03_preprocessing.md")))
    for title, body in main_chunk:
        cells.append(new_code_cell(f"# --- {title} ---\n{body}"))
    cells.append(figure_cell([
        "02_class_distribution.png",
        "03_doc_length_hist.png",
        "03_pos_distribution.png",
        "03_wordcloud.png",
        "03_top_terms.png",
    ]))

    # Section 4: vectorization + search.
    cells.append(new_markdown_cell(read(NARR / "04_vectorization.md")))
    cells.extend(code_cells_from(ROOT / "build" / "section4_vectorization.py"))
    cells.append(figure_cell(["04_tfidf_top_terms.png", "04_tsne.png"]))

    # Section 5: classification + topics.
    cells.append(new_markdown_cell(read(NARR / "05_modeling.md")))
    cells.extend(code_cells_from(ROOT / "build" / "section5_modeling.py"))
    cells.append(figure_cell([
        "05_metrics_comparison.png",
        "05_confusion_matrix_best.png",
        "05_confusion_matrix_multinomialnb.png",
        "05_confusion_matrix_linearsvc.png",
        "05_confusion_matrix_logisticregression.png",
        "05_topics_nmf.png",
    ]))

    # Section 6: NER + knowledge graph.
    cells.append(new_markdown_cell(read(NARR / "06_ner_graph.md")))
    cells.extend(code_cells_from(ROOT / "build" / "section6_ner_graph.py"))
    cells.append(figure_cell([
        "06_entity_types.png",
        "06_top_entities.png",
        "06_knowledge_graph.png",
    ]))
    cells.append(new_markdown_cell(GRAPH_HTML_NOTE))

    # Section 7: synthesis.
    cells.append(new_markdown_cell(read(NARR / "07_synthesis.md")))

    notebook = new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }

    out_path = ROOT / "notebooks" / "pln_pipeline.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, out_path)
    print(f"Wrote {out_path} with {len(cells)} cells "
          f"({sum(c.cell_type == 'code' for c in cells)} code, "
          f"{sum(c.cell_type == 'markdown' for c in cells)} markdown)")


if __name__ == "__main__":
    build()
