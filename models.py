from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from config import MODEL, BASE_URL, LLM_TOKEN
from langchain_openai import ChatOpenAI

from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)


llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=LLM_TOKEN,
    temperature=0,
    max_tokens=2048 * 5,
)

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="ukrhistory11_sparse",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)
