import os
from dotenv import load_dotenv
import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import StuffDocumentsChain
from langchain.llms import HuggingFaceEndpoint

# Load API key from .env
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    st.error("API key not found. Please create a `.env` file with your Hugging Face token.")
    st.stop()

# UI
st.title("📄 Chat with Your PDF")
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf is not None:
    # Extract text
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    # Split text
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([raw_text])

    # Embeddings and vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # LLM and prompt
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=api_token,
        temperature=0.5,
        max_new_tokens=512,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    )

    chain = StuffDocumentsChain(llm_chain=prompt | llm, document_variable_name="context")

    # Ask
    query = st.text_input("Ask a question about the PDF:")
    if query:
        docs = retriever.invoke(query)
        result = chain.invoke({"question": query, "context": docs})
        st.write(result["output"])
