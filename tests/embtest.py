### multilingual-e5-large

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},  # змініть на 'cuda', якщо є відеокарта
    encode_kwargs={"normalize_embeddings": True},  # E5 вимагає нормалізації векторів
)

print(f"Успішно створено {len(docs)} чанків.")


query = "query: Розкажи про базу на Rust"
found_docs = vector_store.similarity_search(query, k=1)

print("\nЗнайдений релевантний шматок тексту:")
# Прибираємо префікс "passage: " при виводі, якщо потрібно
print(found_docs[0].page_content.replace("passage: ", "", 1))
