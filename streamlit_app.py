import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import StuffDocumentsChain

# Ensure event loop is created (for Python >= 3.10)
if sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()

st.set_page_config(page_title="📄 Chat with PDF")
st.title("📄 Chat with PDF")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf is not None:
    # Extract text
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        if page.extract_text():
            raw_text += page.extract_text()

    # Split text into chunks
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = splitter.create_documents([raw_text])

    # Embed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # Hugging Face LLM
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.5,
        max_new_tokens=512,
    )

    # Create prompt and chain (replaces deprecated load_qa_chain)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = StuffDocumentsChain(llm_chain=prompt | llm, document_variable_name="context")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        docs = retriever.invoke(query)  # updated from get_relevant_documents
        answer = chain.invoke({"question": query, "context": docs})  # updated from chain.run
        st.write(answer["output"])
