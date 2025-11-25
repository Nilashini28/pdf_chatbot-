import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -----------------------------------------
# Function to process uploaded PDFs
# -----------------------------------------
def process_pdfs(pdf_files):
    docs = []

    for pdf in pdf_files:
        # Save uploaded file to a temporary file
        temp_path = f"temp_{pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf.getbuffer())

        # Load PDF with loader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        docs.extend(pages)

        # Delete temporary file
        os.remove(temp_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


    # Create FAISS Vector Store
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db


# -----------------------------------------
# Function to get answer from Groq LLM
# -----------------------------------------
def get_answer(query, vector_db):
    # Search for similar chunks
    matched_docs = vector_db.similarity_search(query, k=4)

    # Prepare context
    context = "\n\n".join([doc.page_content for doc in matched_docs])

    # Initialize Groq LLM (FIXED)
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer in simple and clear language:
"""

    response = llm.invoke(prompt)
    return response.content



# -----------------------------------------
# Streamlit UI
# -----------------------------------------
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.header("📚 Chat with Multiple PDFs ")

    # Sidebar: upload PDFs
    with st.sidebar:
        st.subheader("📄 Upload your PDF documents")
        pdf_files = st.file_uploader(
            "Upload PDFs here",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if pdf_files:
                with st.spinner("Processing PDFs..."):
                    st.session_state.vector_store = process_pdfs(pdf_files)
                st.success("PDFs processed successfully! ✔")
            else:
                st.error("Please upload at least one PDF.")

    # Main UI
    query = st.text_input("Ask a question about your documents:")

    if query and "vector_store" in st.session_state:
        with st.spinner("Generating answer..."):
            answer = get_answer(query, st.session_state.vector_store)
        st.write("### 🤖 Answer:")
        st.write(answer)

    elif query:
        st.warning("Please upload and process PDFs first.")


# Run App
if __name__ == "__main__":
    main()
