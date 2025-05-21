import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain

# Fix event loop issue in Streamlit Cloud
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
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = splitter.create_documents([raw_text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.5,
        max_new_tokens=512,
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        docs = retriever.get_relevant_documents(query)
        answer = chain.run(input_documents=docs, question=query)
        st.write(answer)
