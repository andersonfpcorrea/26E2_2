"""Build the frozen preprocessing contract: ``data/processed.parquet``.

This is the single source of truth consumed by every downstream notebook
section (vectorization, classification, NER). Producing it once guarantees all
sections operate on identical cleaned text.

Pipeline: normalize -> tokenize -> drop stopwords (NLTK + custom) -> lemmatize
with spaCy. The canonical ``clean_text`` uses lemmatization (not stemming) to
keep tokens human-readable for search, topics, and graph labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_CSV = DATA_DIR / "bbc_news.csv"
OUTPUT_PARQUET = DATA_DIR / "processed.parquet"

# Use the project-local NLTK data so the build is self-contained.
nltk.data.path.insert(0, str(PROJECT_ROOT / "nltk_data"))

# Domain stopwords: high-frequency reporting verbs and filler that survive
# generic stopword removal but carry no topical signal in news prose.
CUSTOM_STOPWORDS = {
    "said",
    "say",
    "says",
    "mr",
    "mrs",
    "ms",
    "would",
    "could",
    "also",
    "one",
    "two",
    "year",
    "years",
    "told",
    "added",
    "new",
    "make",
    "made",
    "get",
    "go",
    "going",
    "us",
    "uk",
}

TOKEN_PATTERN = re.compile(r"[a-z]+")


def build_stopwords() -> set[str]:
    """Return the union of NLTK English stopwords and domain stopwords."""
    return set(stopwords.words("english")) | CUSTOM_STOPWORDS


def normalize(text: str) -> str:
    """Lowercase and strip everything except ASCII letters and spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_processed() -> pd.DataFrame:
    """Run the full preprocessing pipeline and persist the parquet contract."""
    frame = pd.read_csv(INPUT_CSV)
    stop = build_stopwords()

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    normalized = [normalize(text) for text in frame["text"].tolist()]

    token_lists: list[list[str]] = []
    for doc in nlp.pipe(normalized, batch_size=64):
        tokens = [
            token.lemma_
            for token in doc
            if token.lemma_ not in stop
            and len(token.lemma_) > 2
            and TOKEN_PATTERN.fullmatch(token.lemma_)
        ]
        token_lists.append(tokens)

    frame["clean_text"] = [" ".join(tokens) for tokens in token_lists]
    frame["tokens"] = token_lists
    frame["n_words_raw"] = frame["text"].str.split().apply(len)
    frame["n_words_clean"] = [len(tokens) for tokens in token_lists]

    frame.to_parquet(OUTPUT_PARQUET, index=False)
    return frame


if __name__ == "__main__":
    result = build_processed()
    print(f"Wrote {len(result)} rows to {OUTPUT_PARQUET}")
    print(f"Mean raw words/doc:   {result['n_words_raw'].mean():.1f}")
    print(f"Mean clean words/doc: {result['n_words_clean'].mean():.1f}")
    print(
        f"Vocabulary size (clean): "
        f"{len(set(t for toks in result['tokens'] for t in toks))}"
    )
    print(
        result[["category", "n_words_raw", "n_words_clean"]]
        .groupby("category")
        .mean()
        .round(1)
    )
