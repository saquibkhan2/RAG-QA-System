
import os
from src.document_processing import load_and_chunk_document
from src.vector_store import create_vector_index
from src.retrieval import semantic_search, mmr_rerank
from src.llm_integration import LLMIntegration
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    """
    A complete Retrieval-Augmented Generation (RAG) pipeline for medical question answering.
    """
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', llm_model_name='mistralai/Mistral-7B-Instruct-v0.2'):
        """
        Initializes the RAG pipeline components.

        Args:
            embedding_model_name (str): The name of the sentence transformer model for embeddings.
            llm_model_name (str): The name of the HuggingFace model for the language model.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_integration = LLMIntegration(llm_model=llm_model_name)
        self.index = None
        self.chunks = []
        self.chunk_map = {}
        self.cache = {}

    def load_and_process_document(self, file_path, chunk_size=300, overlap=50):
        """
        Loads a document, chunks it, and creates a FAISS vector index.

        Args:
            file_path (str): The path to the document.
            chunk_size (int): The desired size of each chunk in words.
            overlap (int): The number of words to overlap between chunks.
        """
        self.chunks, self.chunk_map = load_and_chunk_document(file_path, chunk_size, overlap)
        self.index = create_vector_index(self.chunks, model_name=self.embedding_model.model_name)

    def retrieve_context(self, question, k=5, use_mmr=False, lambda_val=0.5):
        """
        Retrieves relevant context chunks for a given question.

        Args:
            question (str): The user's question.
            k (int): The number of top chunks to retrieve.
            use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for reranking.
            lambda_val (float): The lambda value for MMR (balances relevance and diversity).

        Returns:
            list: A list of the most relevant context chunks.
        """
        retrieved_chunks = semantic_search(question, self.index, self.chunks, self.embedding_model, k)
        if use_mmr:
            return mmr_rerank(question, retrieved_chunks, self.embedding_model, lambda_val, k)
        return retrieved_chunks

    def answer_question(self, question, context_chunks):
        """
        Generates an answer to a question using the LLM and context.

        Args:
            question (str): The user's question.
            context_chunks (list): A list of relevant context chunks.

        Returns:
            str: The generated answer.
        """
        # Check cache first
        if (question, tuple(context_chunks)) in self.cache:
            return self.cache[(question, tuple(context_chunks))]

        answer = self.llm_integration.generate_answer(question, context_chunks)
        # Store in cache
        self.cache[(question, tuple(context_chunks))] = answer
        return answer

    def full_pipeline(self, file_path, question, k=5, use_mmr=False, lambda_val=0.5):
        """
        Runs the full RAG pipeline from document loading to answer generation.

        Args:
            file_path (str): The path to the document.
            question (str): The user's question.
            k (int): The number of top chunks to retrieve.
            use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) for reranking.
            lambda_val (float): The lambda value for MMR (balances relevance and diversity).

        Returns:
            tuple: A tuple containing the generated answer and the retrieved context chunks.
        """
        if not self.chunks or not self.index or self.chunk_map.get(0) != os.path.basename(file_path):
            self.load_and_process_document(file_path)

        context_chunks = self.retrieve_context(question, k, use_mmr, lambda_val)
        answer = self.answer_question(question, context_chunks)
        return answer, context_chunks
