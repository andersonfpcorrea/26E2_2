"""Section 5 - Modeling (Requirement 3): supervised classification + topic modeling.

Two complementary fronts over the BBC News corpus:

(A) Supervised classification of ``category`` from TF-IDF features, comparing
    MultinomialNB, LinearSVC and a class-balanced LogisticRegression. The
    balanced logistic regression is the explicit demonstration of how mild
    class imbalance (entertainment 386 vs sport 511) is handled.

(B) Unsupervised topic modeling with both LDA (on raw counts) and NMF (on
    TF-IDF), each with 5 topics to mirror the 5 known editorial sections. The
    discovered topics are inspected against the human categories.

All artifacts (figures, persisted best model) land under ``outputs/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# === CELL: Setup ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DF = pd.read_parquet(PROJECT_ROOT / "data" / "processed.parquet")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TOPICS = 5
TOP_WORDS = 10
CATEGORIES = sorted(DF["category"].unique())


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """Persist ``fig`` to the figures directory with the project conventions."""
    path = FIG_DIR / filename
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# === CELL: Features and split ===
def build_features() -> tuple:
    """Vectorize ``clean_text`` with TF-IDF and produce a stratified split.

    Returns the fitted vectorizer plus the train/test feature matrices and
    label vectors. The split is stratified on ``category`` so that the mild
    class imbalance is preserved identically in train and test.
    """
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(DF["clean_text"])
    labels = DF["category"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return vectorizer, x_train, x_test, y_train, y_test


# === CELL: Train and compare classifiers ===
def build_classifiers() -> dict:
    """Return the three classifiers to compare, keyed by display name."""
    return {
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(dual=True, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_classifiers(x_train, x_test, y_train, y_test) -> tuple:
    """Fit every classifier and collect metrics and predictions.

    Returns ``(models, predictions, comparison)`` where ``comparison`` is a
    DataFrame holding accuracy plus macro precision/recall/F1 per model.
    """
    models = build_classifiers()
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        macro = report["macro avg"]
        rows.append(
            {
                "model": name,
                "accuracy": report["accuracy"],
                "macro_precision": macro["precision"],
                "macro_recall": macro["recall"],
                "macro_f1": macro["f1-score"],
            }
        )

    comparison = pd.DataFrame(rows).set_index("model").round(4)
    return models, predictions, comparison


# === CELL: Confusion matrices ===
def plot_confusion_matrices(y_test, predictions: dict, best_name: str) -> list[Path]:
    """Save a confusion-matrix figure per model plus a dedicated best-model one."""
    paths: list[Path] = []

    for name, y_pred in predictions.items():
        matrix = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
        disp = ConfusionMatrixDisplay(matrix, display_labels=CATEGORIES)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
        ax.set_title(f"Confusion matrix - {name}")
        slug = name.lower()
        paths.append(save_fig(fig, f"05_confusion_matrix_{slug}.png"))

    # Dedicated best-model figure under the contracted filename.
    best_matrix = confusion_matrix(y_test, predictions[best_name], labels=CATEGORIES)
    disp = ConfusionMatrixDisplay(best_matrix, display_labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion matrix - best model ({best_name})")
    paths.append(save_fig(fig, "05_confusion_matrix_best.png"))
    return paths


# === CELL: Metrics comparison chart ===
def plot_metrics_comparison(comparison: pd.DataFrame) -> Path:
    """Save a grouped bar chart of accuracy + macro P/R/F1 across models."""
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    models = comparison.index.tolist()
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, comparison[metric].to_numpy(), width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Classifier comparison (test set)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return save_fig(fig, "05_metrics_comparison.png")


# === CELL: Topic modeling ===
def fit_topic_models() -> tuple:
    """Fit LDA on counts and NMF on TF-IDF, both with 5 topics.

    Returns ``(lda, lda_terms, nmf, nmf_terms)`` where each ``*_terms`` is the
    feature-name array aligned to the model component columns.
    """
    count_vectorizer = CountVectorizer(max_features=20000)
    counts = count_vectorizer.fit_transform(DF["clean_text"])
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, random_state=RANDOM_STATE, learning_method="batch"
    )
    lda.fit(counts)

    tfidf_vectorizer = TfidfVectorizer(max_features=20000)
    tfidf = tfidf_vectorizer.fit_transform(DF["clean_text"])
    nmf = NMF(
        n_components=N_TOPICS, random_state=RANDOM_STATE, init="nndsvda", max_iter=400
    )
    nmf.fit(tfidf)

    return (
        lda,
        np.array(count_vectorizer.get_feature_names_out()),
        nmf,
        np.array(tfidf_vectorizer.get_feature_names_out()),
    )


def top_words_per_topic(
    model, terms: np.ndarray, n: int = TOP_WORDS
) -> list[list[str]]:
    """Return the ``n`` highest-weighted words for each topic component."""
    topics: list[list[str]] = []
    for component in model.components_:
        top_idx = component.argsort()[::-1][:n]
        topics.append(terms[top_idx].tolist())
    return topics


def print_topics(label: str, topics: list[list[str]]) -> None:
    """Print top words per topic for a fitted model."""
    print(f"\n[{label}] top-{TOP_WORDS} words per topic")
    for i, words in enumerate(topics):
        print(f"  Topic {i}: {', '.join(words)}")


def plot_topics(topics: list[list[str]], model, terms: np.ndarray, title: str) -> Path:
    """Save horizontal bar subplots of top words per topic for one model."""
    fig, axes = plt.subplots(1, N_TOPICS, figsize=(4 * N_TOPICS, 5), sharex=False)
    for i, ax in enumerate(axes):
        component = model.components_[i]
        top_idx = component.argsort()[::-1][:TOP_WORDS]
        words = terms[top_idx][::-1]
        weights = component[top_idx][::-1]
        ax.barh(range(len(words)), weights, color="tab:blue")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.set_title(f"Topic {i}")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title)
    return save_fig(fig, "05_topics_nmf.png")


# === CELL: Main ===
def main() -> None:
    """Run both modeling fronts end to end and persist all artifacts."""
    print(f"Corpus: {len(DF)} docs across {len(CATEGORIES)} categories")
    print("Class counts:\n", DF["category"].value_counts().to_string())

    # --- (A) Supervised classification ---
    vectorizer, x_train, x_test, y_train, y_test = build_features()
    print(
        f"\nTF-IDF features: {x_train.shape[1]} | "
        f"train={x_train.shape[0]} test={x_test.shape[0]}"
    )

    models, predictions, comparison = evaluate_classifiers(
        x_train, x_test, y_train, y_test
    )
    print("\n=== Metrics comparison (test set) ===")
    print(comparison.to_string())

    best_name = comparison["macro_f1"].idxmax()
    print(f"\nBest model by macro F1: {best_name}")

    plot_confusion_matrices(y_test, predictions, best_name)
    plot_metrics_comparison(comparison)

    # Persist the best classifier and the vectorizer that produced its features.
    joblib.dump(models[best_name], MODEL_DIR / "clf_best.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "clf_vectorizer.joblib")
    print(f"Saved best model ({best_name}) + vectorizer to {MODEL_DIR}")

    # --- (B) Topic modeling ---
    lda, lda_terms, nmf, nmf_terms = fit_topic_models()
    lda_topics = top_words_per_topic(lda, lda_terms)
    nmf_topics = top_words_per_topic(nmf, nmf_terms)
    print_topics("LDA (counts)", lda_topics)
    print_topics("NMF (TF-IDF)", nmf_topics)

    plot_topics(nmf_topics, nmf, nmf_terms, "NMF topics - top words")

    print("\nDone.")


if __name__ == "__main__":
    main()
