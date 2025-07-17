import streamlit as st
from src.qa_pipeline import RAGPipeline
import os

# Initialize the RAG pipeline
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()

st.title("Medical Document Q&A with RAG")

uploaded_file = st.file_uploader("Upload a medical document (.pdf or .txt)", type=["pdf", "txt"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    if not os.path.exists("temp_docs"):
        os.makedirs("temp_docs")
    file_path = os.path.join("temp_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Load and process the document
    with st.spinner("Processing document..."):
        pipeline.load_and_process_document(file_path)
    st.success("Document processed and indexed!")

    question = st.text_input("Ask a question about the document:")

    # Advanced options
    use_mmr = st.checkbox("Use MMR for diverse retrieval", value=False)
    k_chunks = st.slider("Number of context chunks to retrieve (k)", min_value=1, max_value=10, value=5)

    if st.button("Get Answer"):
        if question:
            with st.spinner("Finding the answer..."):
                answer, context_chunks = pipeline.full_pipeline(file_path, question, k=k_chunks, use_mmr=use_mmr)
                st.write("**Answer:**")
                st.write(answer)

                with st.expander("Show Retrieved Context"):
                    for i, chunk in enumerate(context_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(chunk)
        else:
            st.warning("Please enter a question.")