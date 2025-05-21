import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Replace with your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Ask Questions About Your Document")

uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    file_path = f"temp_file.{uploaded_file.type.split('/')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load the document
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Set up LLM and QA Chain
    llm = OpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Input box
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Generating answer..."):
            result = chain.run(query)
            st.write("### 📌 Answer:")
            st.write(result)
