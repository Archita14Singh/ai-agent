import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import io

# Download NLTK resources
nltk.download("punkt")

# Initialize transformers
summarizer = pipeline("summarization")
qa_pipeline = pipeline("question-answering")

# --- PDF Processing Function ---
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- Split Text into Chunks ---
def split_text(text, max_length=700):
    sentences = text.split('. ')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- Summarization ---
def generate_summary(text):
    chunks = split_text(text)
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summary += result[0]["summary_text"] + " "
    return summary

# --- Q&A ---
def generate_qa(text, question):
    result = qa_pipeline(question=question, context=text)
    return result["answer"]

# --- Mind Map ---
def generate_mind_map(text):
    sentences = nltk.sent_tokenize(text)
    graph = nx.Graph()

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for i in range(len(words) - 1):
            graph.add_edge(words[i], words[i + 1])

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx(graph, with_labels=True, node_size=700, node_color="skyblue", font_size=10, ax=ax)
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("📚 AI Agent on PDF (Q&A, Summarizer & Mind Map)")
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("PDF uploaded and processed!")

    if st.checkbox("🧠 Show Summary"):
        with st.spinner("Summarizing..."):
            summary = generate_summary(pdf_text)
            st.subheader("Summary")
            st.write(summary)

    if st.checkbox("❓ Ask a Question"):
        user_question = st.text_input("Enter your question")
        if user_question:
            answer = generate_qa(pdf_text, user_question)
            st.subheader("Answer")
            st.write(answer)

    if st.checkbox("🗺️ Generate Mind Map"):
        with st.spinner("Generating Mind Map..."):
            generate_mind_map(pdf_text)
