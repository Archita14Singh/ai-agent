import streamlit as st
from transformers import pipeline
import nltk

nltk.download('punkt')

st.set_page_config(page_title="AI Text Assistant", layout="wide")
st.title("AI Text Summarization and Q&A App")

# Cache model loading
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return summarizer, qa

summarizer, qa = load_models()

text_input = st.text_area("Paste your text below:", height=300)

if st.button("Summarize"):
    if text_input.strip():
        with st.spinner("Summarizing..."):
            summary = summarizer(text_input, max_length=130, min_length=30, do_sample=False)
            st.success("Summary:")
            st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text.")

if st.button("Ask a Question"):
    question = st.text_input("What would you like to ask?")
    if text_input.strip() and question.strip():
        with st.spinner("Answering..."):
            answer = qa(question=question, context=text_input)
            st.success("Answer:")
            st.write(answer['answer'])
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        st.warning("Please enter some context text.")
