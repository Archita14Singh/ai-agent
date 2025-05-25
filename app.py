import streamlit as st
from transformers import pipeline
import nltk

# Download nltk tokenizer once
nltk.download('punkt')

st.title("AI Summarization and Q&A App")

# Explicitly specify models to avoid warnings
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")
    return summarizer, qa

summarizer, qa = load_models()

text = st.text_area("Enter text to summarize or ask questions about:")

if st.button("Summarize"):
    if text:
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        st.write("### Summary:")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summarize.")

if st.button("Ask a Question"):
    question = st.text_input("Enter your question:")
    if text and question:
        answer = qa(question=question, context=text)
        st.write("### Answer:")
        st.write(answer['answer'])
    elif not question:
        st.warning("Please enter a question.")
    else:
        st.warning("Please enter text context first.")
