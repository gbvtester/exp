import sys
# print("Python executable:", sys.executable)
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

# --- Section-based Job Spec Parsing ---
import re
from typing import List, Dict

def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    section_pattern = re.compile(r"(\d{6})([\s\S]*?)(end of section)", re.IGNORECASE)
    sections = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        for match in section_pattern.finditer(text):
            section_number = match.group(1)
            section_text = match.group(2).strip()
            # Save section with section number, text, and page number
            sections.append({
                "section_number": section_number,
                "text": section_text,
                "page_number": page_num + 1
            })
    return sections


# Chunking with metadata
def chunk_section(section, chunk_size=500, overlap=50):
    text = section["text"]
    section_number = section["section_number"]
    page_number = section["page_number"]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "section_number": section_number,
            "page_number": page_number
        })
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

# --- Section Filtering UI ---
st.subheader("Section Filtering (Job Spec)")
section_input = st.text_input("Enter section numbers to process (comma separated, e.g. 033000,033100):", "")
selected_sections = [s.strip() for s in section_input.split(",") if s.strip()]

if st.button("Run Semantic Search"):
    prod_path = os.path.join(UPLOAD_DIR, product_file)
    job_path = os.path.join(UPLOAD_DIR, job_file)

    # Product text chunking (no change)
    doc_prod = fitz.open(prod_path)
    prod_text = "".join([page.get_text() for page in doc_prod])
    prod_chunks = []
    prod_chunk_size = 500
    prod_overlap = 50
    start = 0
    while start < len(prod_text):
        end = min(start + prod_chunk_size, len(prod_text))
        prod_chunks.append(prod_text[start:end])
        start += prod_chunk_size - prod_overlap

    # Job spec section parsing and filtering
    all_sections = extract_sections(job_path)
    if selected_sections:
        filtered_sections = [s for s in all_sections if s["section_number"] in selected_sections]
    else:
        filtered_sections = all_sections
    # Chunk filtered sections
    job_chunks_meta = []
    for section in filtered_sections:
        job_chunks_meta.extend(chunk_section(section))
    job_chunks = [c["text"] for c in job_chunks_meta]

    st.write(f"Product chunks: {len(prod_chunks)} | Job spec chunks: {len(job_chunks)} | Sections: {', '.join(selected_sections) if selected_sections else 'All'}")

    # Embedding and FAISS
    job_embeds = embed_chunks(job_chunks)
    faiss_index = build_faiss_index(np.array(job_embeds))
    prod_embeds = embed_chunks(prod_chunks)

    st.subheader("Results")
    for i, (prod_chunk, prod_embed) in enumerate(zip(prod_chunks, prod_embeds)):
        D, I = faiss_index.search(np.array([prod_embed]), k=1)
        match_idx = I[0][0]
        match_meta = job_chunks_meta[match_idx]
        match_text = match_meta["text"]
        section_number = match_meta["section_number"]
        page_number = match_meta["page_number"]
        st.markdown(f"**Feature {i+1}:**")
        st.code(prod_chunk[:300] + ("..." if len(prod_chunk) > 300 else ""))
        st.markdown("**Matching Job Spec Text:**")
        st.code(match_text[:300] + ("..." if len(match_text) > 300 else ""))
        st.write(f"Section: {section_number} | Page: {page_number}")
        st.markdown(f"[Jump to original section] (Section {section_number}, Page {page_number})")
        st.write("---")