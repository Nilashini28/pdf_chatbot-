# 📚 AI PDF Chatbot using Gemini API

An intelligent **multi-PDF chatbot** that allows users to upload multiple PDF documents and ask questions across them.
The application uses **Retrieval-Augmented Generation (RAG)** with **LangChain** and is powered by **Google Gemini 1.5 Pro** for accurate, context-aware answers.

## 🚀 Features

* 📄 Upload and chat with **multiple PDFs simultaneously**
* 🔍 **RAG-based document retrieval** using FAISS
* 🤖 Answers powered by **Gemini 1.5 Pro**
* 🧠 Context-aware responses grounded in document content
* ⚡ Fast and interactive **Streamlit UI**
* ❌ Avoids hallucinations by answering only from retrieved context


## 🧠 Why Gemini API?

* Supports **large context windows**, ideal for multi-PDF reasoning
* Produces **high-quality, clear explanations**
* Works seamlessly with **LangChain RAG pipelines**
* Enables scalable, real-world AI applications


## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini 1.5 Pro
* **Framework:** LangChain
* **Embeddings:** HuggingFace (MiniLM)
* **Vector Database:** FAISS
* **Document Loader:** PyPDFLoader


## 📌 Architecture Overview

1. User uploads multiple PDF files
2. PDFs are split into chunks using LangChain
3. Embeddings are generated and stored in FAISS
4. User query retrieves relevant chunks
5. Gemini API generates an answer using retrieved context

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/pdf-chatbot
cd pdf-chatbot
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set up environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 🎯 Hackathon Theme Alignment

**Theme:** Hyper-Personalized Learning

This application transforms static PDFs into an **interactive learning assistant**, enabling users to:

* Ask questions across multiple documents
* Understand complex content through simplified explanations
* Learn faster with AI-assisted document understanding


## 🧪 Example Use Cases

* Students studying from textbooks or notes
* Researchers analyzing multiple papers
* Professionals reviewing reports or documentation

## 🏆 What Makes This Unique?

* Multi-PDF support with cross-document reasoning
* Gemini-powered responses instead of generic chatbots
* Production-ready RAG pipeline
* Simple yet powerful learning tool

## 📌 Future Enhancements

* Page-level citations
* Quiz and flashcard generation
* Learning level selection (Beginner / Expert)
* Voice-based interaction
* Visual understanding of charts and diagrams

## 📄 License

This project is open-source and available under the MIT License.

## 🙌 Acknowledgements

* Google Gemini API
* LangChain
* Streamlit

