# Updated embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_openai import OpenAIEmbeddings
from typing import Optional
import os

class EmbeddingsManager:
    def __init__(self):
        self.embeddings = None
        
    def init_openai(self, model: str = "text-embedding-3-small", **kwargs):
        self.embeddings = OpenAIEmbeddings(model=model, **kwargs)
        
    def init_local(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # or 'cpu'
            encode_kwargs={'normalize_embeddings': False}
        )
        
    def get_embeddings(self):
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")
        return self.embeddings