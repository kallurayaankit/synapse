# 🧠 Synapse – Personal AI Command Center

[![CI](https://github.com/kallurayaankit/synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/kallurayaankit/synapse/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live-Demo-green?logo=huggingface)](https://kallurayaankit-synapse.hf.space)

**Synapse** is a multi-user AI platform where you can connect your data, build custom AI agents with tools, fine‑tune models on your own documents, and chat with them in real time. It brings together retrieval‑augmented generation (RAG), tool‑calling, fine‑tuning, connectors, and a full observability dashboard—all in one place.

---

## 🚀 Features

- **Multi‑user authentication** with password hashing
- **Document upload** (PDF, TXT, images) and automatic indexing into per‑user Qdrant collections
- **RAG chat** using local LLMs (Mistral via Ollama) with context retrieval
- **Agent configuration** – set a system prompt, enable tools (web search, calculator, email), select a model
- **Fine‑tuning UI** – simulated fine‑tuning jobs that create per‑user models (extensible to real AutoTrain)
- **Connectors** – simulated Google Drive, Notion, GitHub that auto‑ingest sample documents
- **Admin dashboard** – live metrics (requests, tool calls, errors, latency) and recent logs
- **Prometheus metrics** endpoint at `/metrics`

---

## 📁 Project Structure
synapse/
├── app.py # Main FastAPI application (all routes, HTML, logic)
├── requirements.txt # Python dependencies
├── synapse.db # SQLite database (users, configs, models)
├── synapse.log # Application log
├── uploads/ # Temporary upload folder
├── qdrant_storage/ # Qdrant vector data (Docker volume)
└── README.md


---

## ⚡ Quick Start (Local)

### 1. Prerequisites

- [Python 3.12](https://www.python.org/downloads/)
- [Ollama](https://ollama.com) with the **Mistral** model (`ollama pull mistral`)
- [Qdrant](https://qdrant.tech) running locally (`qdrant.exe` or Docker)
- A [Tavily](https://tavily.com) API key for web search (optional)

### 2. Install

```bash
git clone https://github.com/kallurayaankit/synapse.git
cd synapse
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


---

## ⚡ Quick Start (Local)

### 1. Prerequisites

- [Python 3.12](https://www.python.org/downloads/)
- [Ollama](https://ollama.com) with the **Mistral** model (`ollama pull mistral`)
- [Qdrant](https://qdrant.tech) running locally (`qdrant.exe` or Docker)
- A [Tavily](https://tavily.com) API key for web search (optional)

### 2. Install

```bash
git clone https://github.com/kallurayaankit/synapse.git
cd synapse
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

🖥️ Tech Stack

    FastAPI – web framework

    SQLite – user and configuration storage

    Qdrant – vector database for document retrieval

    Sentence‑Transformers – text embeddings

    Ollama – local LLM (Mistral)

    Tavily – web search API

    Pillow / pytesseract – image OCR

    Prometheus‑client – metrics

    PyPDF2 – PDF parsing

📊 Admin Dashboard

The admin dashboard shows:

    Total chat requests

    Tool invocations

    Errors

    Average latency

    Last 100 lines of application logs

Available at /admin after logging in.


---

## 🐙 2. Push to GitHub

### 2.1 Create `.gitignore`

```cmd
echo venv/ > .gitignore
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore
echo uploads/ >> .gitignore
echo qdrant_storage/ >> .gitignore
echo synapse.db >> .gitignore
echo .env >> .gitignore

git init
git add .
git commit -m "Synapse: personal AI command center with RAG, tools, fine-tuning, admin dashboard"
git remote add origin https://github.com/kallurayaankit/synapse.git
git branch -M main
git push -u origin main