"""Sections 2-3 - Corpus inspection and preprocessing (Requirement 1).

Covers loading and inspecting the corpus, word/sentence tokenization,
normalization, stopword removal (NLTK + custom), a fair stemming-vs-
lemmatization vocabulary comparison, spaCy POS tagging, and the visual evidence
(class distribution, document-length histogram, word cloud, top terms, POS
distribution).
"""

# === CELL: Setup and imports ===
import matplotlib

matplotlib.use("Agg")

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud

PROJECT_ROOT = Path(__file__).resolve().parent.parent
nltk.data.path.insert(0, str(PROJECT_ROOT / "nltk_data"))

DF = pd.read_parquet(PROJECT_ROOT / "data" / "processed.parquet")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Domain stopwords: high-frequency reporting verbs and filler that carry no
# topical signal in news prose (mirrors scripts/build_processed.py).
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


# === CELL: Corpus inspection ===
def class_distribution(frame):
    """Return the document count per category, descending."""
    return frame["category"].value_counts()


def plot_class_distribution(frame, path):
    """Save a bar chart of document counts per category."""
    counts = class_distribution(frame)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_title("Document count per category (BBC News)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Documents")
    for i, c in enumerate(counts.values):
        ax.text(i, c, str(c), ha="center", va="bottom")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_length_histogram(frame, path):
    """Save a histogram of raw document length (words) by category."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for category in sorted(frame["category"].unique()):
        subset = frame.loc[frame["category"] == category, "n_words_raw"]
        ax.hist(subset, bins=40, alpha=0.5, label=category)
    ax.set_title("Raw document length distribution by category")
    ax.set_xlabel("Words per document")
    ax.set_ylabel("Documents")
    ax.set_xlim(0, 1200)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# === CELL: Tokenization and normalization demo ===
def normalize(text):
    """Lowercase and strip everything except ASCII letters and spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def demo_tokenization(sample_text):
    """Print sentence and word tokenization of a sample document."""
    sentences = sent_tokenize(sample_text)
    words = word_tokenize(sample_text)
    print(f"Sentences found: {len(sentences)}")
    print(f"First sentence: {sentences[0]}")
    print(f"Word tokens (first 25): {words[:25]}")
    return sentences, words


# === CELL: Stopword removal ===
def build_stopwords():
    """Return the union of NLTK English stopwords and custom domain stopwords."""
    return set(stopwords.words("english")) | CUSTOM_STOPWORDS


def base_tokens(text, stop):
    """Normalize, tokenize, and drop stopwords/short tokens for one document.

    Produces the shared base used for the fair stemming-vs-lemmatization
    comparison.
    """
    tokens = word_tokenize(normalize(text))
    return [t for t in tokens if t.isalpha() and len(t) > 2 and t not in stop]


# === CELL: Stemming vs lemmatization comparison ===
def compare_stemming_lemmatization(frame, stop):
    """Compare vocabulary collapse of stemming vs lemmatization on one base.

    Both Porter stemming and WordNet lemmatization are applied to an identical
    base-token corpus so the comparison isolates the morphological operation.

    Returns a tuple of (summary DataFrame, example DataFrame).
    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    base_vocab = set()
    stem_vocab = set()
    lemma_vocab = set()
    for text in frame["text"]:
        for token in base_tokens(text, stop):
            base_vocab.add(token)
            stem_vocab.add(stemmer.stem(token))
            lemma_vocab.add(lemmatizer.lemmatize(token))

    base_n = len(base_vocab)
    summary = pd.DataFrame(
        {
            "technique": [
                "Base (no stopwords)",
                "Porter stemming",
                "WordNet lemmatization",
            ],
            "vocab_size": [base_n, len(stem_vocab), len(lemma_vocab)],
        }
    )
    summary["reduction_vs_base_%"] = (
        (base_n - summary["vocab_size"]) / base_n * 100
    ).round(1)

    sample_words = [
        "studies",
        "studying",
        "studied",
        "national",
        "nationalisation",
        "running",
        "better",
        "organisation",
        "policies",
        "winning",
        "companies",
        "argued",
    ]
    examples = pd.DataFrame(
        {
            "word": sample_words,
            "porter_stem": [stemmer.stem(w) for w in sample_words],
            "wordnet_lemma": [lemmatizer.lemmatize(w) for w in sample_words],
        }
    )
    return summary, examples


# === CELL: POS tagging with spaCy ===
def pos_distribution(frame, sample_size=500, random_state=42):
    """Compute the coarse POS-tag distribution over a sample using spaCy.

    A sample keeps the demonstration fast; the distribution is stable across
    samples for a corpus this homogeneous.
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    sample = frame.sample(min(sample_size, len(frame)), random_state=random_state)
    counter = Counter()
    for doc in nlp.pipe(sample["text"].tolist(), batch_size=64):
        for token in doc:
            if not token.is_space and not token.is_punct:
                counter[token.pos_] += 1
    return counter


def demo_pos_tagging(sample_sentence):
    """Print the POS tags of a sample sentence with spaCy."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    doc = nlp(sample_sentence)
    pairs = [(token.text, token.pos_, token.tag_) for token in doc][:15]
    for text, pos, tag in pairs:
        print(f"  {text:<15} {pos:<8} {tag}")
    return pairs


def plot_pos_distribution(counter, path):
    """Save a bar chart of the POS-tag distribution."""
    items = counter.most_common()
    tags = [t for t, _ in items]
    counts = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(tags, counts, color="#55A868")
    ax.set_title("POS-tag distribution (spaCy, sampled documents)")
    ax.set_xlabel("Universal POS tag")
    ax.set_ylabel("Token count")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# === CELL: Word cloud and top terms ===
def plot_wordcloud(frame, path):
    """Save a word cloud of the cleaned corpus text."""
    text = " ".join(frame["clean_text"].fillna(""))
    cloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=150,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word cloud of cleaned BBC corpus")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def top_terms(frame, top_n=25):
    """Return the most frequent cleaned tokens across the corpus."""
    counter = Counter()
    for tokens in frame["tokens"]:
        counter.update(tokens)
    return counter.most_common(top_n)


def plot_top_terms(frame, path, top_n=25):
    """Save a horizontal bar chart of the most frequent cleaned tokens."""
    items = top_terms(frame, top_n)
    terms = [t for t, _ in items][::-1]
    counts = [c for _, c in items][::-1]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(terms, counts, color="#C44E52")
    ax.set_title(f"Top {top_n} terms in cleaned corpus")
    ax.set_xlabel("Frequency")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# === CELL: Main ===
def main():
    """Run inspection and preprocessing evidence end to end."""
    print(f"Corpus shape: {DF.shape}")
    print("\nClass distribution:\n", class_distribution(DF).to_string())
    print(f"\nMean raw words/doc:   {DF['n_words_raw'].mean():.1f}")
    print(f"Mean clean words/doc: {DF['n_words_clean'].mean():.1f}")
    print("\nLength by category (raw words):")
    print(
        DF.groupby("category")["n_words_raw"]
        .describe()[["mean", "50%", "std"]]
        .round(1)
        .to_string()
    )

    plot_class_distribution(DF, FIG_DIR / "02_class_distribution.png")
    plot_length_histogram(DF, FIG_DIR / "03_doc_length_hist.png")

    print("\n=== Tokenization demo ===")
    demo_tokenization(DF.iloc[0]["text"])

    stop = build_stopwords()
    print(
        f"\nStopwords: {len(stopwords.words('english'))} NLTK + "
        f"{len(CUSTOM_STOPWORDS)} custom = {len(stop)} total"
    )

    print("\n=== Stemming vs lemmatization ===")
    summary, examples = compare_stemming_lemmatization(DF, stop)
    print(summary.to_string(index=False))
    print("\nExamples:")
    print(examples.to_string(index=False))

    print("\n=== POS tagging demo ===")
    demo_pos_tagging(sent_tokenize(DF.iloc[0]["text"])[0])
    pos_counter = pos_distribution(DF)
    print("\nPOS distribution (sample):")
    for tag, count in pos_counter.most_common():
        print(f"  {tag:<8} {count}")
    plot_pos_distribution(pos_counter, FIG_DIR / "03_pos_distribution.png")

    plot_wordcloud(DF, FIG_DIR / "03_wordcloud.png")
    plot_top_terms(DF, FIG_DIR / "03_top_terms.png")
    print("\nSection 2-3 figures written.")


if __name__ == "__main__":
    main()
