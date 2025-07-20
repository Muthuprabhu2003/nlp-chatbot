from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import os
import streamlit as st

# Set HuggingFace API Token (optional, but required for some models)
api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_key"

def load_pdf_chunks(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

def answer_question(vectorstore, question):
    docs = vectorstore.similarity_search(question)
    chain = load_qa_chain(HuggingFaceHub(repo_id="google/flan-t5-base"), chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    return answer
 