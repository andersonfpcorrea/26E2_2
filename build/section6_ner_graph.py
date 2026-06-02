"""Section 6 - NER, information extraction and knowledge graph (Requirement 4).

Runs spaCy NER over the BBC corpus, extracts structured patterns with regex
(dates, money, percentages, emails, URLs), normalizes near-duplicate entity
surface forms with rapidfuzz, builds a NetworkX co-occurrence knowledge graph,
computes degree and betweenness centrality, renders static (matplotlib) and
interactive (PyVis) visualizations, exports GraphML, and answers an analytical
question about the most central entities in BBC news coverage.
"""

# === CELL: Setup and imports ===
import matplotlib

matplotlib.use("Agg")

import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import pandas as pd
import spacy
from pyvis.network import Network
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
nltk.data.path.insert(0, str(PROJECT_ROOT / "nltk_data"))

DF = pd.read_parquet(PROJECT_ROOT / "data" / "processed.parquet")
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
GRAPH_DIR = PROJECT_ROOT / "outputs" / "graph"
FIG_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
RAW_TEXTS = DF["text"].fillna("").tolist()

# Entity labels we keep for the knowledge graph (named, content-bearing types).
GRAPH_LABELS = {"PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "EVENT"}


# === CELL: spaCy NER over the corpus ===
def run_ner(texts, batch_size=64):
    """Run spaCy NER over the raw corpus and return per-document entities.

    The parser, tagger and lemmatizer are disabled for speed; tok2vec and ner
    are kept because the NER component depends on token vectors.

    Args:
        texts: Iterable of raw document strings.
        batch_size: Batch size passed to ``nlp.pipe``.

    Returns:
        List of per-document lists of ``(entity_text, label)`` tuples.
    """
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["parser", "tagger", "lemmatizer", "attribute_ruler"],
    )
    doc_entities = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        ents = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
        doc_entities.append(ents)
    return doc_entities


def aggregate_entities(doc_entities):
    """Aggregate entity counts by label and by surface form per label.

    Args:
        doc_entities: Output of :func:`run_ner`.

    Returns:
        Tuple of (label_counter, per_label_surface_counters, overall_counter).
    """
    label_counter = Counter()
    per_label = defaultdict(Counter)
    overall = Counter()
    for ents in doc_entities:
        for text, label in ents:
            label_counter[label] += 1
            per_label[label][text] += 1
            overall[text] += 1
    return label_counter, per_label, overall


def plot_entity_types(label_counter, path):
    """Save a bar chart of entity-label frequencies."""
    labels, counts = zip(*label_counter.most_common())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, counts, color="#4C72B0")
    ax.set_title("Entity frequency by spaCy label (BBC corpus)")
    ax.set_xlabel("Entity label")
    ax.set_ylabel("Mentions")
    plt.xticks(rotation=45, ha="right")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_top_entities(per_label, key_labels, path, top_n=10):
    """Save a grid of bar charts with the top entities for key labels."""
    n = len(key_labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, key_labels):
        items = per_label[label].most_common(top_n)
        names = [name for name, _ in items][::-1]
        vals = [cnt for _, cnt in items][::-1]
        ax.barh(names, vals, color="#55A868")
        ax.set_title(f"Top {label}")
        ax.set_xlabel("Mentions")
    fig.suptitle("Most frequent entities per key type (BBC corpus)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# === CELL: Regex pattern extraction ===
# Money: optional currency symbol/word + number + optional magnitude suffix.
MONEY_RE = re.compile(
    r"(?:[$£€]|\bUS\$)\s?\d[\d,]*(?:\.\d+)?\s?(?:bn|billion|m|million|k|trillion|tn)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|pounds|euros|pence|cents)\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(
    r"\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?(?:per\s?cent|percent)\b", re.IGNORECASE
)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s)?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s?(?:\d{1,2})?,?\s?\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s,)]+", re.IGNORECASE)

REGEX_PATTERNS = {
    "money": MONEY_RE,
    "percentage": PERCENT_RE,
    "date": DATE_RE,
    "email": EMAIL_RE,
    "url": URL_RE,
}


def extract_patterns(texts):
    """Extract regex-based structured fields from the corpus.

    Args:
        texts: Iterable of raw document strings.

    Returns:
        Dict mapping pattern name to a list of all matched strings.
    """
    matches = {name: [] for name in REGEX_PATTERNS}
    for text in texts:
        for name, pattern in REGEX_PATTERNS.items():
            matches[name].extend(m.strip() for m in pattern.findall(text))
    return matches


# === CELL: Fuzzy normalization of entity surface forms ===
def _normalize_key(name):
    """Lower-case a surface form and strip common honorifics/suffixes."""
    cleaned = re.sub(
        r"\b(mr|mrs|ms|dr|sir|the|inc|corp|ltd|plc|co|group|"
        r"company|corporation)\b\.?",
        "",
        name.lower(),
    )
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_entities(overall_counter, threshold=88):
    """Cluster near-duplicate surface forms into canonical entities.

    Surface forms are sorted by frequency; each new form is matched against
    existing cluster representatives using token-sort fuzzy ratio (after a
    light honorific/suffix normalization). Forms above ``threshold`` collapse
    into the most frequent representative.

    Args:
        overall_counter: Counter of surface form -> mention count.
        threshold: Minimum rapidfuzz ratio (0-100) to merge two forms.

    Returns:
        Tuple of (surface_to_canonical map, canonical_counter, n_merged) where
        n_merged is the number of surface forms that collapsed into another.
    """
    forms = [f for f, _ in overall_counter.most_common()]
    surface_to_canonical = {}
    canonical_counter = Counter()
    reps = []  # list of (canonical_surface, normalized_key)

    n_merged = 0
    for form in forms:
        key = _normalize_key(form)
        if not key:
            continue
        best_rep = None
        best_score = 0.0
        for rep_form, rep_key in reps:
            score = fuzz.token_sort_ratio(key, rep_key)
            if score > best_score:
                best_score, best_rep = score, rep_form
        if best_rep is not None and best_score >= threshold:
            surface_to_canonical[form] = best_rep
            canonical_counter[best_rep] += overall_counter[form]
            n_merged += 1
        else:
            surface_to_canonical[form] = form
            canonical_counter[form] += overall_counter[form]
            reps.append((form, key))
    return surface_to_canonical, canonical_counter, n_merged


# === CELL: Knowledge graph from entity co-occurrence ===
def build_graph(
    doc_entities, surface_to_canonical, canonical_counter, top_n=55, min_edge_weight=3
):
    """Build a co-occurrence knowledge graph of the most frequent entities.

    Nodes are the ``top_n`` most frequent canonical entities (restricted to
    graph-relevant labels). An edge connects two entities that appear in the
    same document; the edge weight is the number of co-occurring documents.
    Edges below ``min_edge_weight`` are dropped to keep the graph readable.

    Args:
        doc_entities: Output of :func:`run_ner`.
        surface_to_canonical: Map produced by :func:`normalize_entities`.
        canonical_counter: Canonical entity -> total mention count.
        top_n: Number of nodes to keep.
        min_edge_weight: Minimum co-occurrence count for an edge to be kept.

    Returns:
        A weighted, undirected :class:`networkx.Graph`.
    """
    # Determine the canonical label of each canonical entity (majority vote).
    label_votes = defaultdict(Counter)
    for ents in doc_entities:
        for text, label in ents:
            canon = surface_to_canonical.get(text)
            if canon is not None and label in GRAPH_LABELS:
                label_votes[canon][label] += 1

    eligible = [c for c in canonical_counter if label_votes.get(c)]
    eligible.sort(key=lambda c: canonical_counter[c], reverse=True)
    top_entities = eligible[:top_n]
    top_set = set(top_entities)

    edge_weights = Counter()
    for ents in doc_entities:
        present = set()
        for text, label in ents:
            canon = surface_to_canonical.get(text)
            if canon in top_set and label in GRAPH_LABELS:
                present.add(canon)
        for a, b in combinations(sorted(present), 2):
            edge_weights[(a, b)] += 1

    graph = nx.Graph()
    for ent in top_entities:
        top_label = label_votes[ent].most_common(1)[0][0]
        graph.add_node(ent, label=top_label, mentions=canonical_counter[ent])
    for (a, b), w in edge_weights.items():
        if w >= min_edge_weight:
            graph.add_edge(a, b, weight=w)
    return graph


# === CELL: Centrality ===
def compute_centrality(graph):
    """Compute degree and betweenness centrality.

    Args:
        graph: A NetworkX graph.

    Returns:
        Tuple of (degree_centrality dict, betweenness_centrality dict).
    """
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, weight="weight", seed=RANDOM_STATE)
    return degree, betweenness


# === CELL: Visualization ===
def plot_graph_static(graph, degree, path):
    """Render the knowledge graph with matplotlib, sizing nodes by centrality."""
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(graph, k=0.6, weight="weight", seed=RANDOM_STATE)
    sizes = [3000 * degree.get(n, 0) + 120 for n in graph.nodes()]

    labels = sorted({graph.nodes[n]["label"] for n in graph.nodes()})
    palette = plt.cm.tab10.colors
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}
    node_colors = [color_map[graph.nodes[n]["label"]] for n in graph.nodes()]
    edge_widths = [0.3 + 0.25 * graph[u][v]["weight"] for u, v in graph.edges()]

    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.25, ax=ax)
    nx.draw_networkx_nodes(
        graph, pos, node_size=sizes, node_color=node_colors, alpha=0.9, ax=ax
    )
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=lab,
            markerfacecolor=color_map[lab],
            markersize=10,
        )
        for lab in labels
    ]
    ax.legend(handles=handles, title="Entity type", loc="upper right")
    ax.set_title(
        "BBC knowledge graph - entity co-occurrence (node size = degree centrality)"
    )
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_graph_interactive(graph, degree, path):
    """Render an interactive PyVis HTML visualization of the graph."""
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )
    labels = sorted({graph.nodes[n]["label"] for n in graph.nodes()})
    palette = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
    ]
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}
    for n in graph.nodes():
        lab = graph.nodes[n]["label"]
        net.add_node(
            n,
            label=n,
            color=color_map[lab],
            size=10 + 60 * degree.get(n, 0),
            title=f"{n} ({lab}) - {graph.nodes[n]['mentions']} mentions",
        )
    for u, v in graph.edges():
        w = graph[u][v]["weight"]
        net.add_edge(u, v, value=w, title=f"co-occurs in {w} docs")
    net.force_atlas_2based()
    net.save_graph(str(path))


# === CELL: Reporting helpers ===
def top_by_value(d, top_n=10):
    """Return the top ``top_n`` (key, value) pairs sorted descending by value."""
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


# === CELL: Main pipeline ===
def main():
    """Run the full NER and knowledge-graph pipeline and print a report."""
    print(f"Corpus: {len(RAW_TEXTS)} documents")

    print("Running spaCy NER ...")
    doc_entities = run_ner(RAW_TEXTS)
    label_counter, per_label, overall = aggregate_entities(doc_entities)

    print("\n=== Entity-label distribution ===")
    for label, count in label_counter.most_common():
        print(f"{label:>8}: {count:,}")

    print("\n=== Top 10 entities overall ===")
    for name, count in overall.most_common(10):
        print(f"{count:>5}  {name}")

    key_labels = ["PERSON", "ORG", "GPE"]
    plot_entity_types(label_counter, FIG_DIR / "06_entity_types.png")
    plot_top_entities(per_label, key_labels, FIG_DIR / "06_top_entities.png")

    print("\n=== Regex pattern extraction ===")
    matches = extract_patterns(RAW_TEXTS)
    for name, vals in matches.items():
        examples = [v for v in vals if v][:3]
        print(f"{name:>11}: {len(vals):,} matches | examples: {examples}")

    print("\n=== Fuzzy normalization ===")
    surface_to_canonical, canonical_counter, n_merged = normalize_entities(overall)
    print(f"Distinct surface forms : {len(overall):,}")
    print(f"Canonical entities     : {len(canonical_counter):,}")
    print(f"Surface forms merged   : {n_merged:,}")

    print("\nBuilding knowledge graph ...")
    graph = build_graph(doc_entities, surface_to_canonical, canonical_counter)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    assert graph.number_of_nodes() >= 20, "Graph must have at least 20 nodes"

    degree, betweenness = compute_centrality(graph)
    print("\n=== Top 10 by degree centrality ===")
    for name, val in top_by_value(degree):
        print(f"{val:.4f}  {name}")
    print("\n=== Top 10 by betweenness centrality ===")
    for name, val in top_by_value(betweenness):
        print(f"{val:.4f}  {name}")

    plot_graph_static(graph, degree, FIG_DIR / "06_knowledge_graph.png")
    plot_graph_interactive(graph, degree, GRAPH_DIR / "knowledge_graph.html")
    nx.write_graphml(graph, GRAPH_DIR / "knowledge_graph.graphml")

    print("\nArtifacts written:")
    for p in [
        FIG_DIR / "06_entity_types.png",
        FIG_DIR / "06_top_entities.png",
        FIG_DIR / "06_knowledge_graph.png",
        GRAPH_DIR / "knowledge_graph.html",
        GRAPH_DIR / "knowledge_graph.graphml",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
