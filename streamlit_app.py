import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Title
st.set_page_config(page_title="📄 Chat with your PDF")
st.title("📄 Chat with your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    try:
        # Read PDF
        reader = PdfReader(uploaded_file)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([full_text])

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever()

        # Load LLM
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
            model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
        )

        # Retrieval Chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # User input
        user_input = st.text_input("Ask a question about the document:")
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    result = qa.run(user_input)
                    st.success("Answer:")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error during question answering: {e}")
    except Exception as e:
        st.error(f"PDF processing error: {e}")
