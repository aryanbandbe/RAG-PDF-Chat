# 🧠 Chat with Your PDF (RAG App)

This is a Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, and Ollama's TinyLLaMA model. It allows users to upload any PDF and ask questions about its contents — getting intelligent, contextual answers.

## 🚀 How It Works

1. **Upload PDF**  
   The app extracts and chunks the document text into manageable pieces.

2. **Text Embedding**  
   Each chunk is embedded using a SentenceTransformer (`all-MiniLM-L6-v2`).

3. **Vector Storage (FAISS)**  
   The chunks are indexed with FAISS for fast semantic retrieval.

4. **Question Answering**  
   When the user asks a question, the most relevant chunk is retrieved and sent to the local TinyLLaMA model via a Flask API using Ollama.

5. **Answer Displayed**  
   The model's answer is shown in the Streamlit app interface.

---

## 🛠️ Technologies Used

- **Streamlit** – for the interactive web UI
- **LangChain** – for retrieval pipeline
- **FAISS** – for semantic vector search
- **sentence-transformers** – to embed document chunks
- **Ollama** – to run TinyLLaMA locally
- **Flask** – simple backend to connect Streamlit with Ollama

---

## 📦 Setup Instructions (for local development)

1
git clone https://github.com/YOUR_USERNAME/RAG-PDF-Chat.git
cd RAG-PDF-Chat

2.
conda create -n ragpdf python=3.10 -y
conda activate ragpdf

3.
pip install -r requirements.txt


4.
ollama run tinyllama

5.
python app.py

6.
streamlit run streamlit_app.py --server.runOnSave false


