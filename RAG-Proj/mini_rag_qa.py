import os
import fitz  # PyMuPDF
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
import sys

load_dotenv()
PPLX_API_KEY = os.getenv('PERPLEXITY_API_KEY')
assert PPLX_API_KEY, 'Set your PERPLEXITY_API_KEY in the .env file!'

# --- Document Ingestion and Chunking ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_into_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(sentences, chunk_size=300, chunk_overlap=50):
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        word_count = len(sentence.split())
        if current_chunk_size + word_count > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Overlap
            overlap = []
            overlap_size = 0
            j = len(current_chunk) - 1
            while j >= 0 and overlap_size < chunk_overlap:
                overlap.insert(0, current_chunk[j])
                overlap_size += len(current_chunk[j].split())
                j -= 1
            current_chunk = overlap
            current_chunk_size = overlap_size
        current_chunk.append(sentence)
        current_chunk_size += word_count
        i += 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# --- Embedding and Vector Store ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def embed_chunks(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]

# --- LLM Integration (Perplexity API, Sonar-pro) ---
def ask_perplexity(query, context_chunks):
    context = '\n\n'.join([c for c, _ in context_chunks])
    prompt = f'''You are a helpful medical assistant. Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find a specific answer to this question in the provided context."

Context:
{context}

Question: {query}'''
    headers = {
        'Authorization': f'Bearer {PPLX_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'sonar-pro',
        'messages': [{"role": "user", "content": prompt}]
    }
    resp = requests.post('https://api.perplexity.ai/chat/completions', headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()['choices'][0]['message']['content'].strip()
    else:
        return f'Perplexity API error: {resp.text}'

# --- Main Script ---
def main():
    # You can change this path or use sys.argv[1]
    pdf_path = 'documents/9241544228_eng.pdf'
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    print(f'Loading and processing: {pdf_path}')
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    chunks = chunk_text(sentences)
    print(f'Chunked into {len(chunks)} chunks.')
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    # Sample questions
    questions = [
        "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission",
        "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?"
    ]
    for q in questions:
        print(f'\nQ: {q}')
        context_chunks = retrieve(q, index, chunks, k=3)
        print('Top context:')
        for i, (ctx, score) in enumerate(context_chunks):
            print(f'  [{i+1}] (score={score:.3f}) {ctx[:120]}...')
        answer = ask_perplexity(q, context_chunks)
        print(f'\nA: {answer}\n')

    # Allow user to ask their own question
    while True:
        user_q = input("\nEnter your own question (or 'exit' to quit): ").strip()
        if user_q.lower() == 'exit':
            break
        context_chunks = retrieve(user_q, index, chunks, k=3)
        print('Top context:')
        for i, (ctx, score) in enumerate(context_chunks):
            print(f'  [{i+1}] (score={score:.3f}) {ctx[:120]}...')
        answer = ask_perplexity(user_q, context_chunks)
        print(f'\nA: {answer}\n')

if __name__ == '__main__':
    main() 