import streamlit as st
from transformers import pipeline
import nltk

# Download required NLTK data
nltk.download('punkt')

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Streamlit app layout
st.set_page_config(page_title="AI Agent", layout="centered")
st.title("🤖 AI Agent - Question Answering")

# Input area
context = st.text_area("Enter context paragraph:", height=200)
question = st.text_input("Ask a question related to the context:")

# Button to get answer
if st.button("Get Answer"):
    if context.strip() and question.strip():
        with st.spinner("Thinking..."):
            result = qa_pipeline({
                'context': context,
                'question': question
            })
            st.success(f"📝 Answer: {result['answer']}")
    else:
        st.warning("⚠️ Please enter both context and question.")
