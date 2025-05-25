import streamlit as st
from transformers import pipeline
import nltk

# Download necessary NLTK data (optional, depending on your use case)
nltk.download('punkt')

# Load the model (question answering as an example)
qa_pipeline = pipeline("question-answering")

st.title("AI Agent Demo")

context = st.text_area("Enter some context:")
question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline({
            'context': context,
            'question': question
        })
        st.success(f"Answer: {result['answer']}")
    else:
        st.warning("Please enter both context and question.")
