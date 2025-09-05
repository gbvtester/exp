import streamlit as st
import os
import shutil
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Local PDF Semantic Pipeline")

# --- File Upload ---
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success("Files uploaded!")

# --- File Manager ---
st.subheader("File Manager")
files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
delete_file = st.selectbox("Select a file to delete", [""] + files)
if delete_file and st.button("Delete Selected File"):
    os.remove(os.path.join(UPLOAD_DIR, delete_file))
    st.success(f"Deleted {delete_file}")
    st.experimental_rerun()
st.write("Uploaded PDFs:", files)

# --- Select Product Datasheet and Job Spec ---
st.subheader("Select PDFs for Processing")
if len(files) >= 2:
    product_file = st.selectbox("Product Datasheet", files, key="prod")
    job_file = st.selectbox("Job Spec", files, key="job")
else:
    st.info("Upload at least 2 PDFs to proceed.")
    st.stop()

# --- PDF Text Extraction ---
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Embedding and FAISS Search ---
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()

def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- Processing ---
if st.button("Run Semantic Search"):
    prod_path = os.path.join(UPLOAD_DIR, product_file)
    job_path = os.path.join(UPLOAD_DIR, job_file)
    prod_text = extract_text(prod_path)
    job_text = extract_text(job_path)

    prod_chunks = chunk_text(prod_text)
    job_chunks = chunk_text(job_text)

    st.write(f"Product chunks: {len(prod_chunks)} | Job spec chunks: {len(job_chunks)}")

    job_embeds = embed_chunks(job_chunks)
    faiss_index = build_faiss_index(np.array(job_embeds))

    prod_embeds = embed_chunks(prod_chunks)

    st.subheader("Results")
    for i, (prod_chunk, prod_embed) in enumerate(zip(prod_chunks, prod_embeds)):
        D, I = faiss_index.search(np.array([prod_embed]), k=1)
        match_idx = I[0][0]
        match_text = job_chunks[match_idx]
        st.markdown(f"**Feature {i+1}:**")
        st.code(prod_chunk[:300] + ("..." if len(prod_chunk) > 300 else ""))
        st.markdown("**Matching Job Spec Text:**")
        st.code(match_text[:300] + ("..." if len(match_text) > 300 else ""))
        st.write("---")