# PDFQueryBot
A Smart Q&A Chatbot for PDF Understanding using LangChain + HuggingFace

**DocuMind** is an AI-powered Question Answering (Q&A) chatbot that reads a PDF, understands its content, and answers user queries in natural language. This project is built using **LangChain**, **Hugging Face LLMs**, and **ChromaDB** — enabling semantic search and natural language responses over static PDF files.

Use cases include:
- Customer support automation for document-heavy industries
- Legal/medical document Q&A
- Academic research assistance

---

## Problem Statement

Traditional PDFs are static — to find anything, you must *search manually*. This project converts such documents into **interactive, conversational sources of knowledge**.

---

## Tools & Technologies Used

- **LangChain** – For chaining prompts and managing retrieval-based QA logic  
- **Hugging Face Transformers** – Language models for understanding and generating responses  
- **Chroma DB** – Lightweight vector store for fast and efficient semantic retrieval  
- **PyPDF2** – For reading and extracting text from PDFs  
- **RecursiveCharacterTextSplitter** – For intelligent document chunking  
- **PromptTemplate, LLMChain** – For modular, maintainable prompting  

---

## How It Works

1. **PDF Ingestion**: Extracts text from the uploaded PDF using `PyPDF2`.  
2. **Text Chunking**: Splits long documents into manageable pieces using `RecursiveCharacterTextSplitter`.  
3. **Embedding Generation**: Converts chunks into embeddings using `HuggingFaceEmbeddings`.  
4. **Vector Store Creation**: Stores these embeddings in `Chroma` vector DB.  
5. **User Query Processing**: Takes the user's question, retrieves relevant chunks, and passes them to an LLM (Hugging Face endpoint).  
6. **Response Generation**: Outputs a clean, context-aware answer.  

---

## Challenges Faced & How I Solved Them

### 🔸 Inconsistent Model Answers  
- **Problem**: Model hallucinated or gave vague answers.  
- **Solution**: Tuned `chunk_size` and `overlap` for better context + restructured prompt template.  

### 🔸 High Latency  
- **Problem**: Large models slowed response time.  
- **Solution**: Switched to GPU-accelerated endpoints from Hugging Face's inference API.  

### 🔸 Chaining Complexity  
- **Problem**: Prompt logic became hard to manage.  
- **Solution**: Used `PromptTemplate` and `LLMChain` for cleaner code structure.  

---

## Why This Approach?

- **LangChain** provides a scalable way to manage retrieval + LLM logic  
- **ChromaDB** is fast, lightweight, and easy to set up for prototyping  
- **Hugging Face LLMs** are more customizable and cost-effective for experimentation  
- This stack supports **modularity**, **explainability**, and **rapid development**

---

## What’s Next?

- Integrating **Gradio** or **Streamlit** for a simple user interface  
- Enhancing with **RAG (Retrieval-Augmented Generation)** for better factual grounding  
- Adding support for **multiple PDFs** and multi-language documents  

