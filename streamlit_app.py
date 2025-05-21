import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceEndpoint
from huggingface_hub import InferenceClient
import os

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_token"

# Initialize the InferenceClient with the specific task
client = InferenceClient(model="gpt2", token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

# Initialize the HuggingFaceEndpoint with the specified task
llm = HuggingFaceEndpoint(
    repo_id="gpt2",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    task="text-generation"
)

st.title("PDF Question Answering App")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Load and process the PDF
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create a FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create a retriever
    retriever = vectorstore.as_retriever()

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User input for questions
    question = st.text_input("Ask a question about the PDF:")

    if question:
        answer = qa_chain.run(question)
        st.write("Answer:", answer)
