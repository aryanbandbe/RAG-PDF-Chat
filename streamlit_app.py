import os
import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import StuffDocumentsChain
from langchain.llms import HuggingFaceHub

# Load .env variables
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_token:
    st.error("❌ API token not found. Please add it to a `.env` file.")
    st.stop()

st.title("📄 Chat with Your PDF")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf is not None:
    # Step 1: Extract text
    reader = PdfReader(pdf)
    raw_text = ""
    for page in reader.pages:
        if text := page.extract_text():
            raw_text += text

    # Step 2: Split text into chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([raw_text])

    # Step 3: Create embeddings and FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # Step 4: Setup LLM using HuggingFaceHub (NOT HuggingFaceEndpoint)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=api_token
    )

    # Step 5: Create a prompt and chain
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

    # Step 6: User Query
    query = st.text_input("Ask a question about the PDF:")
    if query:
        docs = retriever.invoke(query)
        response = chain.invoke({"question": query, "context": docs})
        st.write(response["output"])
