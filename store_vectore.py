import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from io import BytesIO
import json

PDF_DIRECTORY = "/Users/tusharsharma/Documents/Ayurvedic ChatBot/data"  
VECTORSTORE_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.json"

def get_single_pdf_chunks(pdf, text_splitter):
    pdf_bytes = BytesIO(pdf.read())
    pdf_reader = PdfReader(pdf_bytes)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks

def get_all_pdfs_chunks(pdf_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )

    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

def save_vectorstore(vectorstore, texts):
    # Save FAISS index
    vectorstore.save_local(VECTORSTORE_INDEX_PATH)
    
    # Save metadata (texts) separately as JSON
    with open(METADATA_PATH, "w") as file:
        json.dump(texts, file)

def load_pdfs_and_build_vectorstore():
    # Load PDFs from the directory
    pdf_docs = []
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            with open(pdf_path, "rb") as pdf_file:
                pdf_docs.append(BytesIO(pdf_file.read()))

    # Process PDF text into chunks
    text_chunks = get_all_pdfs_chunks(pdf_docs)
    
    # Create the vectorstore with the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    # Save vectorstore to disk
    save_vectorstore(vectorstore, text_chunks)

if __name__ == "__main__":
    load_pdfs_and_build_vectorstore()
