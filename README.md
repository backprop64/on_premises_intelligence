# On-Premises Intelligence (OPI) - find and chat though your personal documents

**Local, private Retrieval-Augmented Generation for personal data analysis.**

Imagine having a smart assistant that *could* comb through your private collection of documents — anything from confidential contracts to specialised medical guidelines or complex tax records. By keeping everything on **your own machine**, OPI is designed to index those files securely and let you ask plain-English questions such as *"Which clause covers early termination?"* or *"What's the deductible for procedure 99214?"*. It then attempts to surface the most relevant passages and draft a concise answer — all **without sending any data to the cloud**. Potential benefits:

* **Privacy by default** – sensitive data never leaves your firewall.
* **Faster insight** – cut hours of scrolling through PDFs down to seconds.
* **Domain flexibility** – use for legal, financial, medical, engineering and other private documents.


---

## Key Features

* **100 % on-prem** – your documents never leave the device.
* **RAG + Smart File Search** – ask questions *or* just search: the sidebar becomes a live file-system search as you type.
* **Multimodel-ready** – resaons about text and image data, powered by open source LMs and VLMs.
* **One-page UI** – chat, upload, and document browser all in a single responsive page.
* **FastAPI + SSE** – answers stream token-by-token for instant feedback.


---

## 🚀 Quickstart

The project is **Python 3.10 or 3.11** compatible. The snippet below shows the
recommended workflow with **conda** (feel free to use `venv`/`pipenv` if you
prefer).

```bash
# 1. Create and activate an isolated environment
conda create -n opi python=3.11 -y
conda activate opi

# 2. Install the Python packages
pip install --upgrade pip wheel
pip install -r requirements.txt

# 3. Fire it up
./start_chat.sh
```

The helper script will:

1. start the FastAPI server on **http://localhost:8000**;
2. launch a tiny static file server on **http://localhost:5173**; and
3. open your default browser at **`frontend/chat.html`** (the single-page app).

---

## Repository Layout (TL;DR)

```
api/                 # FastAPI application
frontend/            # Static web UI (single-page chat + upload)
opi_file_system/     # Uploaded files are stored here
src/                 # RAG pipeline, database interface, models
config.py            # All user-configurable parameters
start_chat.sh        # helper script launch backend + frontend
requirements.txt     # Python dependencies
```
---
