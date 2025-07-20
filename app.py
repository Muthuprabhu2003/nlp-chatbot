import streamlit as st
from chatbot import load_pdf_chunks, create_vector_store, answer_question

st.set_page_config(page_title="Local Document Q&A Bot")
st.title("📄 Local PDF Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        chunks = load_pdf_chunks(uploaded_file)
        vectorstore = create_vector_store(chunks)
        st.success("Document ready. Ask a question!")

        question = st.text_input("Ask a question from this PDF:")
        if question:
            with st.spinner("Thinking..."):
                answer = answer_question(vectorstore, question)
                st.write("### 🤖 Answer:")
                st.write(answer)
