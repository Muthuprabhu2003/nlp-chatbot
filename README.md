This is a simple but powerful chatbot that can **answer questions based on the content of any uploaded PDF document** using Natural Language Processing (NLP). It leverages `LangChain`, `HuggingFace`, and `FAISS` to provide context-aware responses.

## 🚀 Features

- 📁 Upload any local PDF file (e.g., syllabus, resumes, reports)
- 💬 Ask natural language questions about the document
- 🧠 Powered by `LangChain`, `FAISS`, and `HuggingFace` models
- 🖥️ Clean, modern Streamlit UI
- ☁️ Ready for Streamlit Cloud deployment

## 🧰 Tech Stack

- Python
- Streamlit (UI)
- LangChain (QA pipeline)
- HuggingFace Embeddings (`google/flan-t5-base`)
- FAISS (Vector similarity search)
- PyPDF2 (PDF parser)

## 🛠️ How to Run Locally

1. **Clone this repo**:
   ```bash
   git remote add origin https://github.com/Muthuprabhu2003/nlp-chatbot.git
   cd nlp-chatbot
