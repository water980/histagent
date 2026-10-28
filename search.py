from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

vectorstore = QdrantVectorStore.from_existing_collection(
    collection_name="ukrhistory11_sparse",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)


results = vectorstore.similarity_search(
    "розкажи про Мануїльського",
    k=5,
)
print(
    "розкажи про Мануїльського",
    end="\n\n\n"
)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 80)




results = vectorstore.similarity_search(
    "Мануїльский",
    k=5,
)
print(
    "Мануїльский",
    end="\n\n\n"
)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 80)