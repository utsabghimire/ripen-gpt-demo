# RipenGPT Demo

This is a small demonstration of an **AI‑assisted research platform** designed for
fruit development, ripening and postharvest senescence.  The app is intended
to showcase how domain knowledge, ontologies and multi‑omics datasets can be
integrated into a single interactive tool.  Although this prototype is not
connected to any external LLM, GitHub or database, the structure makes it
easy to extend.

## Features

- **Overview tab** – A brief description of the project and its goals.
- **Paper summaries** – Summaries of three published studies on broccoli
  senescence and postharvest treatments.  Each summary is distilled from
  published abstracts and author notes; no copyrighted text is included.
- **Document Q/A** – A simple retrieval system built on top of a
  TF‑IDF vectorizer.  Users can type a question and receive a
  citation‑backed answer drawn from the summary corpus.  Queries that request
  wet‑lab protocols or step‑by‑step instructions are blocked.
- **Gene Explorer** – An interactive table of example gene expression values
  across several storage conditions.  You can search for a gene of interest
  and view a bar chart of its relative expression.
- **Ontology filter** – A panel showing a minimal plant ontology (species,
  tissues and developmental stages) used by the retrieval system.  In a
  full implementation these tags would be used to filter and re‑rank
  search results.
- **Proteomics** – A small table of protein abundance changes (log₂ fold
  change) measured during postharvest senescence.
- **GitHub tab** – A placeholder tab illustrating where code from a connected
  GitHub repository could be surfaced.  Since the API for the user’s
  repositories is not accessed in this prototype, a short explanation is
  provided instead.

## Running the App

The application is written with [Streamlit](https://streamlit.io/) and
requires Python 3.8 or later.  To install the required dependencies and run
the app:

```bash
cd ripen_gpt_demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run ripen_gpt_app.py
```

If you do not wish to install the optional sentence‑transformer library,
the TF‑IDF retriever will still function.  See `requirements.txt` for
details.
