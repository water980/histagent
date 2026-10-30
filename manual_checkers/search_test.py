from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from sentence_transformers import CrossEncoder

COLLECTION_NAME = "ukrhistory11_sparse"


# Dense embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# Sparse embeddings (BM25)
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)


# Hybrid Qdrant
vectorstore = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)


reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

def rerank(query, documents, limit=5):
    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    return ranked[:limit]

def search(text, limit=5):
    candidates = vectorstore.similarity_search(
        text,
        k=20,
    )

    # 2. Reranking
    ranked = rerank(
        text,
        candidates,
        limit=limit,
    )

    print(f"\n🔎 Запит: {text}\n")

    for i, doc in enumerate(ranked, 1):
        print(f"#{i}")
        # print(doc[1])
        print(doc[1].page_content.replace("passage: ", "", 1))

        # print(f"Metadata: {doc.metadata}")    
        print("=" * 80)


while True:
    text = input("\n🔎 Введи текст (або exit): ")

    if text.lower() == "exit":
        break

    search(text)