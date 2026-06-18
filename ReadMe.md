# Zyro HR Assistant

Production RAG system for answering Zyro Dynamics HR policy questions.

## Features

- Hybrid Retrieval (FAISS + BM25)
- BGE Large Embeddings
- Cross Encoder Reranking
- Llama 3.3 70B via Groq
- Source Grounding
- Out-of-Scope Detection

## Run

pip install -r requirements.txt

streamlit run app.py