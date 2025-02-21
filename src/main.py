from llm_setup import LLMClient
from dotenv import load_dotenv
from embeddings import EmbeddingsManager

from langchain_community.document_loaders import GitLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import subprocess

load_dotenv()

def setup_qa_system():
    # Define repository URL and local path
    repo_url = "https://github.com/veniceai/api-docs.git"
    local_path = "./veniceai-api-docs"
    
    # Clone the repository if it doesn't exist locally
    if not os.path.exists(local_path):
        subprocess.run(["git", "clone", repo_url, local_path])
    
    # Set up the GitLoader to load specific file types from the local repo
    loader = GitLoader(
        repo_path=local_path,
        branch="main",  # Specify the branch (e.g., "main")
        file_filter=lambda file_path: file_path.endswith((".md", ".mdx", ".rst", ".txt")),
    )
    
    # Define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    
    # Load and split the documents using load_and_split
    chunks = loader.load_and_split(text_splitter)
    
    # Initialize embeddings
    embeddings_manager = EmbeddingsManager()
    embeddings_manager.init_local()  # Switch to init_openai() for OpenAI embeddings
    embeddings = embeddings_manager.get_embeddings()
    
    # Create FAISS vector store and retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Initialize Venice API client and LLM
    client = LLMClient()
    llm = client.init_venice()  # Uses environment variables for configuration
    
    # Set up the QA chain
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

if __name__ == "__main__":
    try:
        qa_chain = setup_qa_system()
    except Exception as e:
        print(f"Error setting up QA system: {e}")
        exit()

    print("System ready! Type /exit to quit")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == "/exit":
            break
        
        try:
            answer = qa_chain.invoke(question)
            print('\nAnswer:')
            print(answer['result'])
        except Exception as e:
            print(f"Error getting answer: {e}")