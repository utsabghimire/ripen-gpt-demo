"""
RipenGPT Demo Application

This Streamlit app demonstrates a miniature research assistant for fruit
development, ripening and postharvest senescence.  It is intentionally
lightweight and self‑contained: rather than connecting to external large
language models or proprietary databases, it reads a few text summaries,
performs simple vector‑space retrieval and displays example multi‑omics
data.  The structure is modular so that additional data sources, models
and APIs can be integrated easily.

Running the app requires the libraries listed in ``requirements.txt``.

To start the app:

```
streamlit run ripen_gpt_app.py
```

"""

import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_documents() -> List[dict]:
    """Load summary documents from the sample_corpus folder.

    Each document is stored as a plain text file with header metadata
    (title, authors, year, citation) followed by a free‑text summary.  The
    file name (without extension) is used as an identifier.

    Returns:
        A list of dicts with keys: id, title, authors, year, citation, text.
    """
    docs = []
    corpus_dir = os.path.join(os.path.dirname(__file__), "sample_corpus")
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(corpus_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Simple parsing of header lines
        lines = content.splitlines()
        metadata = {"id": fname, "title": None, "authors": None, "year": None, "citation": None}
        summary_lines = []
        for line in lines:
            if line.lower().startswith("title:"):
                metadata["title"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("authors:"):
                metadata["authors"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("year:"):
                metadata["year"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("citation:"):
                metadata["citation"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:"):
                # summary heading – skip
                continue
            else:
                summary_lines.append(line)
        metadata["text"] = "\n".join(summary_lines).strip()
        # Fallback for missing title
        if not metadata["title"]:
            metadata["title"] = fname.replace(".txt", "").replace("_", " ").title()
        docs.append(metadata)
    return docs


@st.cache_data(show_spinner=False)
def build_vectorizer(docs: List[dict]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Create a TF‑IDF vectorizer and document matrix.

    Args:
        docs: list of document dicts containing a 'text' key.

    Returns:
        (vectorizer, matrix) tuple.
    """
    corpus = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


@st.cache_data(show_spinner=False)
def load_gene_data() -> pd.DataFrame:
    """Load example gene expression data from CSV."""
    path = os.path.join(os.path.dirname(__file__), "data", "gene_expression.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_proteomics_data() -> pd.DataFrame:
    """Load example proteomics data from CSV."""
    path = os.path.join(os.path.dirname(__file__), "data", "proteomics.csv")
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_ontology() -> dict:
    """Load the minimal plant ontology from JSON."""
    path = os.path.join(os.path.dirname(__file__), "ontology", "plant_ontology_min.json")
    with open(path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    return ontology

# -----------------------------------------------------------------------------
# Text processing helpers
# -----------------------------------------------------------------------------

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Return a naive summary consisting of the first few sentences.

    This helper splits text on period characters.  It does not perform
    sophisticated sentence segmentation but suffices for short summaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text
    summary = " ".join(sentences[:max_sentences])
    return summary


def retrieve_documents(query: str, docs: List[dict], vectorizer: TfidfVectorizer, tfidf_matrix: np.ndarray, topn: int = 3) -> List[Tuple[dict, float]]:
    """Retrieve documents most similar to the query using cosine similarity.

    Args:
        query: user question
        docs: list of document dicts
        vectorizer: fitted TfidfVectorizer
        tfidf_matrix: document TF‑IDF matrix
        topn: number of documents to return

    Returns:
        List of tuples (doc, score) sorted by decreasing similarity.
    """
    if not query:
        return []
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Get top N indices
    top_indices = np.argsort(scores)[::-1][:topn]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        results.append((docs[idx], float(scores[idx])))
    return results


def is_disallowed_query(query: str) -> bool:
    """Check whether a query contains disallowed intent.

    We block queries that explicitly ask for step‑by‑step experimental
    protocols, troubleshooting, CRISPR design or similar wet‑lab instructions.
    The check is conservative and looks for certain keywords.
    """
    banned_keywords = [
        "protocol", "step", "how to", "transform", "transformation", "edit", "troubleshoot",
        "CRISPR", "genome editing", "cas9", "procedures", "instructions"
    ]
    q = query.lower()
    return any(keyword in q for keyword in banned_keywords)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    # Load data
    docs = load_documents()
    vectorizer, tfidf_matrix = build_vectorizer(docs)
    gene_df = load_gene_data()
    prot_df = load_proteomics_data()
    ontology = load_ontology()

    # Page config
    st.set_page_config(page_title="RipenGPT Demo", layout="wide")
    st.title("RipenGPT Demo: Broccoli Senescence Knowledge Hub")

    # Tabs for different sections
    tab_titles = [
        "Overview",
        "Paper Summaries",
        "Document Q/A",
        "Gene Explorer",
        "Proteomics",
        "Ontology",
        "GitHub"
    ]
    tabs = st.tabs(tab_titles)

    # -------------------------------------------------------------------------
    # Overview
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.header("Overview")
        st.markdown(
            """
            **RipenGPT** is a research assistant prototype for exploring gene
            regulation, ripening and senescence in horticultural crops.  The
            system combines domain knowledge, ontologies and small datasets to
            provide concise answers to conceptual questions.  While this demo
            uses only TF‑IDF based retrieval and a handful of broccoli‐specific
            summaries, it illustrates how an interactive tool could accelerate
            literature triage, hypothesis generation and data exploration in an
            academic lab or institute.

            **Key objectives:**
            
            - Provide evidence‑backed answers with citations to the source
              documents.
            - Offer interactive exploration of gene expression and proteomics
              data across different postharvest conditions.
            - Demonstrate the use of a simple plant ontology to filter
              information by species, tissue or developmental stage.
            - Suggest how integration with GitHub and other data sources could
              further enrich the platform.
            """
        )

        st.subheader("Getting Started")
        st.markdown(
            """
            Use the navigation tabs above to explore the different features.  The
            **Paper Summaries** tab contains short, hand‑written summaries of
            three published articles on broccoli senescence.  The **Document
            Q/A** tab lets you type a question and receive a citation‑backed
            answer.  In **Gene Explorer** and **Proteomics** you can browse
            example multi‑omics datasets, and the **Ontology** tab shows the
            minimal vocabulary used for filtering.  Finally, the **GitHub**
            tab explains how repository integration could work in a future
            version.
            """
        )

    # -------------------------------------------------------------------------
    # Paper Summaries
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.header("Paper Summaries")
        st.write(
            "The summaries below are derived from published abstracts and author notes.  They "
            "highlight key findings without reproducing copyrighted text.  Each summary "
            "includes the citation for further reading."
        )
        for doc in docs:
            with st.expander(doc["title"], expanded=False):
                st.markdown(f"**Authors:** {doc.get('authors', 'N/A')}")
                st.markdown(f"**Year:** {doc.get('year', 'N/A')}")
                st.markdown(f"**Citation:** {doc.get('citation', 'N/A')}")
                st.markdown(f"**Summary:**\n\n{doc['text']}")

    # -------------------------------------------------------------------------
    # Document Q/A
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.header("Ask a question")
        st.markdown(
            """
            Enter a conceptual question about broccoli senescence, ripening or
            postharvest treatments.  The system will search the summaries above
            and return the most relevant passages.  Queries that request
            procedural details or wet‑lab instructions are not permitted.
            """
        )
        query = st.text_input("Your question", "")
        if query:
            if is_disallowed_query(query):
                st.error(
                    "Sorry, this prototype does not provide step‑by‑step protocols, "
                    "genome editing instructions or troubleshooting guidance.  Please ask a "
                    "high‑level conceptual question instead."
                )
            else:
                results = retrieve_documents(query, docs, vectorizer, tfidf_matrix, topn=3)
                if not results:
                    st.warning("No relevant documents found.  Try a different question.")
                else:
                    st.success(f"Showing {len(results)} result(s).")
                    for doc, score in results:
                        summary = summarize_text(doc["text"], max_sentences=2)
                        st.markdown(f"### {doc['title']}")
                        st.markdown(f"**Score:** {score:.3f}")
                        st.markdown(f"**Citation:** {doc.get('citation', 'N/A')}")
                        st.markdown(f"**Answer:** {summary}")

    # -------------------------------------------------------------------------
    # Gene Explorer
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.header("Gene Expression Explorer")
        st.write(
            "This table contains example relative expression values for selected genes "
            "across several storage conditions.  The numbers are synthetic and are "
            "intended solely for demonstration.  Enter a gene name to view a bar chart."
        )
        gene_input = st.text_input("Gene name (e.g., SAG12)")
        # Display full table with ability to scroll
        st.dataframe(gene_df, use_container_width=True)
        if gene_input:
            gene_input_upper = gene_input.strip().upper()
            row = gene_df[gene_df["gene"].str.upper() == gene_input_upper]
            if row.empty:
                st.warning(f"Gene '{gene_input}' not found in the table.")
            else:
                st.subheader(f"Expression profile for {gene_input_upper}")
                # Drop the gene column and transpose for plotting
                values = row.drop(columns=["gene"]).T
                values.columns = [gene_input_upper]
                st.bar_chart(values)

    # -------------------------------------------------------------------------
    # Proteomics
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.header("Proteomics Changes")
        st.write(
            "Log₂ fold changes of protein abundance during postharvest senescence.  "
            "Negative values indicate proteins that decrease during storage, while "
            "positive values indicate proteins that accumulate."
        )
        st.dataframe(prot_df, use_container_width=True)
        # Plot bar chart of proteomics data
        chart_data = prot_df.set_index("protein")
        st.bar_chart(chart_data)

    # -------------------------------------------------------------------------
    # Ontology
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.header("Plant Ontology")
        st.write(
            "This simplified ontology lists species, tissues and developmental stages used by "
            "the retrieval system.  In a full implementation, these categories would be used "
            "to filter and boost search results."
        )
        species_keys = list(ontology.get("species", {}).keys())
        tissue_keys = list(ontology.get("tissues", {}).keys())
        stage_keys = list(ontology.get("stages", {}).keys())
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_species = st.multiselect("Species", species_keys, default=["broccoli"])
        with col2:
            selected_tissues = st.multiselect("Tissues", tissue_keys)
        with col3:
            selected_stages = st.multiselect("Stages", stage_keys)
        st.subheader("Selected category synonyms")
        if selected_species:
            st.markdown("**Species synonyms:**")
            st.json({sp: ontology["species"][sp] for sp in selected_species})
        if selected_tissues:
            st.markdown("**Tissue synonyms:**")
            st.json({t: ontology["tissues"][t] for t in selected_tissues})
        if selected_stages:
            st.markdown("**Stage synonyms:**")
            st.json({s: ontology["stages"][s] for s in selected_stages})

    # -------------------------------------------------------------------------
    # GitHub
    # -------------------------------------------------------------------------
    with tabs[6]:
        st.header("GitHub Integration (Placeholder)")
        st.write(
            "In a production deployment, this tab would connect to your GitHub "
            "repository to fetch code, datasets or notebooks relevant to fruit senescence. "
            "For example, the app could display recent commits, load sequence data or "
            "render analysis notebooks.  The current demo does not access any external "
            "services, so this area contains only explanatory text."
        )
        st.markdown(
            """
            ### Example of what could be displayed

            ```python
            # This code would run on the server to fetch a file from GitHub via the API
            from github import Github

            def fetch_notebook(repo_name: str, file_path: str, token: str):
                g = Github(token)
                repo = g.get_repo(repo_name)
                file_content = repo.get_contents(file_path)
                return file_content.decoded_content.decode('utf-8')

            # Example usage:
            # notebook = fetch_notebook('username/ripen-project', 'notebooks/analysis.ipynb', GITHUB_TOKEN)
            ```

            The above snippet illustrates how the [PyGithub](https://pygithub.readthedocs.io/)
            library could be used to retrieve a notebook from a private repository,
            given an access token.  In an interactive app, the notebook could be
            displayed or executed on demand.
            """
        )


if __name__ == "__main__":
    main()