from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)


qdrant_url = "http://localhost:6333"

client = QdrantClient(url=qdrant_url)

with open("histbooktext.txt", "r", encoding="utf-8") as f:

    raw_text = f.read()


text_spltr = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

texts = text_spltr.split_text(raw_text)

docs = [
    Document(
        page_content=f"passage: {chunk}",
    )

    for chunk in texts
]

print(f"Успішно створено {len(docs)} чанків.")

vector_store = QdrantVectorStore.from_documents(
    docs,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    collection_name="ukrhistory11_sparse",
    url=qdrant_url,
)
