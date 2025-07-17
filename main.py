

import os
from src.qa_pipeline import RAGPipeline

if __name__ == '__main__':
    pipeline = RAGPipeline()
    file_path = 'RAG-Proj/documents/9241544228_eng.pdf'

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
    else:
        # Load and process the document once
        print(f"Loading and processing document: {file_path}")
        pipeline.load_and_process_document(file_path)
        print("Document processed and indexed.")

        # Example questions
        question1 = "Give me the correct coded classification for the following diagnosis: 'Recurrent depressive disorder, currently in remission'"
        question2 = "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?"

        # Run the pipeline for the first question
        print(f"\nQuestion: {question1}")
        answer1, context1 = pipeline.full_pipeline(file_path, question1, use_mmr=True)
        print(f"Answer: {answer1}")
        print("Retrieved Context Chunks:")
        for i, chunk in enumerate(context1):
            print(f"  Chunk {i+1}: {chunk[:100]}...") # Print first 100 chars of chunk

        # Run the pipeline for the second question
        print(f"\nQuestion: {question2}")
        answer2, context2 = pipeline.full_pipeline(file_path, question2, use_mmr=True)
        print(f"Answer: {answer2}")
        print("Retrieved Context Chunks:")
        for i, chunk in enumerate(context2):
            print(f"  Chunk {i+1}: {chunk[:100]}...") # Print first 100 chars of chunk

