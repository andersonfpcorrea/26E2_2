"""Section 4 - Vector representation and textual search (Requirement 2).

Implements Bag-of-Words, TF-IDF (with bigrams), a TF-IDF cosine-similarity
search engine, Word2Vec embeddings, and a t-SNE visualization of the corpus.
"""

# === CELL: Setup and imports ===
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DF = pd.read_parquet(PROJECT_ROOT / "data" / "processed.parquet")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CLEAN_TEXT = DF["clean_text"].fillna("").tolist()


# === CELL: Bag-of-Words (CountVectorizer) ===
def build_bag_of_words(documents):
    """Fit a CountVectorizer on the documents and return the vectorizer and matrix.

    Args:
        documents: Iterable of normalized text strings.

    Returns:
        Tuple of (fitted CountVectorizer, sparse document-term matrix).
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def top_terms_by_count(vectorizer, matrix, top_n=20):
    """Return the most frequent terms across the whole corpus.

    Args:
        vectorizer: Fitted CountVectorizer.
        matrix: Document-term count matrix.
        top_n: Number of terms to return.

    Returns:
        List of (term, total_count) tuples sorted descending by count.
    """
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = counts.argsort()[::-1][:top_n]
    return [(terms[i], int(counts[i])) for i in order]


bow_vectorizer, bow_matrix = build_bag_of_words(CLEAN_TEXT)
bow_vocab_size = len(bow_vectorizer.vocabulary_)
bow_top_terms = top_terms_by_count(bow_vectorizer, bow_matrix, top_n=20)

print("=== Bag-of-Words ===")
print(f"BoW matrix shape: {bow_matrix.shape}")
print(f"BoW vocabulary size: {bow_vocab_size}")
print("Top 20 BoW terms (by total count):")
for term, count in bow_top_terms:
    print(f"  {term}: {count}")


# === CELL: TF-IDF with n-grams ===
def build_tfidf(documents, ngram_range=(1, 2), min_df=2, max_df=0.9):
    """Fit a TfidfVectorizer (unigrams + bigrams) and return vectorizer and matrix.

    Args:
        documents: Iterable of normalized text strings.
        ngram_range: N-gram range passed to TfidfVectorizer.
        min_df: Minimum document frequency for a term to be kept.
        max_df: Maximum document frequency (proportion) for a term.

    Returns:
        Tuple of (fitted TfidfVectorizer, sparse TF-IDF matrix).
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def top_terms_by_mean_tfidf(vectorizer, matrix, top_n=20):
    """Return terms ranked by their mean TF-IDF weight across the corpus.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        matrix: TF-IDF document-term matrix.
        top_n: Number of terms to return.

    Returns:
        List of (term, mean_tfidf) tuples sorted descending.
    """
    mean_weights = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = mean_weights.argsort()[::-1][:top_n]
    return [(terms[i], float(mean_weights[i])) for i in order]


tfidf_vectorizer, tfidf_matrix = build_tfidf(CLEAN_TEXT)
tfidf_vocab_size = len(tfidf_vectorizer.vocabulary_)
tfidf_top_terms = top_terms_by_mean_tfidf(tfidf_vectorizer, tfidf_matrix, top_n=20)

print("\n=== TF-IDF (ngram_range=(1,2)) ===")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"TF-IDF vocabulary size: {tfidf_vocab_size}")
print("Top 20 TF-IDF terms (by mean weight):")
for term, weight in tfidf_top_terms:
    print(f"  {term}: {weight:.5f}")


# === CELL: TF-IDF top-terms bar chart ===
def plot_top_tfidf_terms(top_terms, out_path):
    """Save a horizontal bar chart of the top mean TF-IDF terms.

    Args:
        top_terms: List of (term, mean_tfidf) tuples.
        out_path: Destination path for the PNG figure.
    """
    terms = [t for t, _ in top_terms][::-1]
    weights = [w for _, w in top_terms][::-1]
    plt.figure(figsize=(9, 8))
    plt.barh(terms, weights, color="#3b7dd8")
    plt.xlabel("Mean TF-IDF weight")
    plt.title("Top terms by mean TF-IDF (unigrams + bigrams)")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


plot_top_tfidf_terms(tfidf_top_terms, FIG_DIR / "04_tfidf_top_terms.png")


# === CELL: TF-IDF cosine-similarity search engine ===
def search(query, top_n=5):
    """Search the corpus for documents most similar to a free-text query.

    The query is projected into the fitted TF-IDF space and ranked by cosine
    similarity against every document in the corpus.

    Args:
        query: Free-text search string.
        top_n: Number of top results to return.

    Returns:
        List of dicts with keys id, category, title, score, ordered by score.
    """
    query_vector = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).ravel()
    top_idx = scores.argsort()[::-1][:top_n]
    results = []
    for idx in top_idx:
        row = DF.iloc[idx]
        results.append(
            {
                "id": int(row["id"]),
                "category": str(row["category"]),
                "title": str(row["title"]),
                "score": float(scores[idx]),
            }
        )
    return results


SEARCH_QUERIES = [
    "stock market interest rates and economy",
    "football championship final match",
    "new smartphone mobile technology",
]

print("\n=== TF-IDF Search Engine ===")
for query in SEARCH_QUERIES:
    print(f'\nQuery: "{query}"')
    for rank, hit in enumerate(search(query, top_n=5), start=1):
        print(
            f"  {rank}. [{hit['category']}] {hit['title']} "
            f"(score={hit['score']:.4f})"
        )


# === CELL: Word2Vec embeddings ===
def train_word2vec(token_lists, vector_size=100, window=5, min_count=5, seed=42):
    """Train a Word2Vec model on tokenized documents.

    Args:
        token_lists: Iterable of token lists (one per document).
        vector_size: Embedding dimensionality.
        window: Context window size.
        min_count: Minimum token frequency to be included in the vocabulary.
        seed: Random seed for reproducibility.

    Returns:
        Trained gensim Word2Vec model.
    """
    sentences = [list(tokens) for tokens in token_lists]
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        seed=seed,
        workers=1,
    )


w2v_model = train_word2vec(DF["tokens"].tolist())
w2v_model.save(str(MODEL_DIR / "word2vec.model"))

SEED_WORDS = ["government", "film", "market"]
print("\n=== Word2Vec most_similar ===")
print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
for word in SEED_WORDS:
    if word in w2v_model.wv:
        similar = w2v_model.wv.most_similar(word, topn=5)
        formatted = ", ".join(f"{w} ({s:.3f})" for w, s in similar)
        print(f"  {word}: {formatted}")
    else:
        print(f"  {word}: not in vocabulary")


# === CELL: t-SNE visualization ===
def build_tsne_embedding(matrix, n_svd=50, random_state=42):
    """Reduce a high-dimensional sparse matrix to 2D via TruncatedSVD then t-SNE.

    Args:
        matrix: Sparse document-term matrix (e.g. TF-IDF).
        n_svd: Intermediate dimensionality for TruncatedSVD (capped below n_features).
        random_state: Random seed for reproducibility.

    Returns:
        NumPy array of shape (n_documents, 2) with the t-SNE coordinates.
    """
    n_components_svd = min(n_svd, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components_svd, random_state=random_state)
    reduced = svd.fit_transform(matrix)
    tsne = TSNE(n_components=2, random_state=random_state, init="pca")
    return tsne.fit_transform(reduced)


def plot_tsne(coords, categories, out_path):
    """Save a 2D scatter plot of t-SNE coordinates colored by category.

    Args:
        coords: Array of shape (n_documents, 2) with t-SNE coordinates.
        categories: Sequence of category labels per document.
        out_path: Destination path for the PNG figure.
    """
    categories = pd.Series(categories)
    plt.figure(figsize=(10, 8))
    palette = plt.cm.tab10.colors
    for i, cat in enumerate(sorted(categories.unique())):
        mask = (categories == cat).to_numpy()
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.6,
            color=palette[i % len(palette)],
            label=cat,
        )
    plt.legend(title="Category", markerscale=2)
    plt.title("t-SNE of TF-IDF document vectors (SVD-50 -> t-SNE)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


tsne_coords = build_tsne_embedding(tfidf_matrix, n_svd=50, random_state=RANDOM_STATE)
plot_tsne(tsne_coords, DF["category"].tolist(), FIG_DIR / "04_tsne.png")
print("\nt-SNE figure saved.")


# === CELL: Persist artifacts ===
joblib.dump(
    {"vectorizer": tfidf_vectorizer, "matrix": tfidf_matrix},
    MODEL_DIR / "tfidf_vectorizer.joblib",
)
print("TF-IDF vectorizer and matrix persisted.")
print("Section 4 complete.")
