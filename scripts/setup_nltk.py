"""Download the NLTK datasets required by the pipeline into a project-local dir.

The data lands in ``nltk_data/`` (git-ignored, ~117 MB) so the corpus build and
the notebook are self-contained without polluting the user's home directory.
"""

from __future__ import annotations

from pathlib import Path

import nltk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "nltk_data"

PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def setup() -> None:
    """Download every required NLTK package into the project-local directory."""
    TARGET.mkdir(parents=True, exist_ok=True)
    for package in PACKAGES:
        ok = nltk.download(package, download_dir=str(TARGET), quiet=True)
        print(f"{'OK ' if ok else 'FAIL':<5} {package}")


if __name__ == "__main__":
    setup()
