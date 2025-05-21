import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama



st.set_page_config(page_title="RAG Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF using RAG + Ollama")

uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

if uploaded_file:
    try:
        # Read PDF text
        reader = PdfReader(uploaded_file)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if not full_text.strip():
            st.error("❌ Could not extract any text from the PDF. Try another document.")
            st.stop()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([full_text])
        if not docs:
            st.error("❌ No text chunks created. Please upload a valid PDF.")
            st.stop()

        # Embed and build vector store
        with st.spinner("🔍 Creating vector store..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)
            retriever = db.as_retriever()

        # Load Ollama model
        llm = Ollama(model="tinyllama")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Input and output
        user_input = st.text_input("🧠 Ask a question about the document:")
        if user_input:
            with st.spinner("🤖 Thinking..."):
                result = qa.run(user_input)
                st.success("✅ Answer:")
                st.write(result)

    except Exception as e:
        st.error(f"🚨 An error occurred:\n\n{e}")
