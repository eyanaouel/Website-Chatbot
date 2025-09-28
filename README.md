# TEK-UP Website Chatbot

![Chatbot](https://img.shields.io/badge/status-active-success)

## Overview
This project implements an AI-powered chatbot for the TEK-UP University website.  
It leverages **Large Language Models** (LLaMa, Gemma, Qwen) through **Groq API**, along with **LangChain**, **FAISS**, and **HuggingFace embeddings** for RAG (Retrieval-Augmented Generation).  
The chatbot can answer user questions accurately using a pre-populated vector database of website content.

---

## Features

- **Multimodel Support:** LLaMa 3.3 (70B), Gemma 2 (9B), LLaMa 3 (8B), Qwen (32B)
- **RAG Integration:** Retrieve relevant documents for precise answers
- **Streamlit Interface:** Friendly and interactive UI
- **Vector Database:** Preprocessed website content using FAISS
- **Custom Context Handling:** Context-aware responses using retrieved documents
- **Web Crawling & Embeddings:** Crawl TEK-UP website and populate FAISS DB

---

## File Structure

- `chatbot.py` - Streamlit interface for the TEK-UP chatbot
- `craxler.py` - Web crawler to extract website content and populate FAISS database
- `vdb/` - Local FAISS vector database
- `.env` - Stores `GROQ_API_KEY` (not tracked in Git)
- `requirements.txt` - Python dependencies

---

## Setup

1. Clone the repository:
```bash
git clone https://github.com/eyanaouel/TEKUP_Chatbot.git
cd TEKUP_Chatbot
