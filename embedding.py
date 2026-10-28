### multilingual-e5-large

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda'},  # змініть на 'cuda', якщо є відеокарта
    encode_kwargs={'normalize_embeddings': True}  # E5 вимагає нормалізації векторів
)
sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)

# with open("histbooktext.txt", "r", encoding="utf-8") as f:
    

qdrant_url = "http://localhost:6333" 

client = QdrantClient(url=qdrant_url)


# with open("histbooktext.txt", "r", encoding="utf-8") as f:
with open("histbooktext.txt", "r", encoding="utf-8") as f:

    raw_text = f.read()


text_spltr = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    length_function=len
)

texts = text_spltr.split_text(raw_text)

# 2. Підготовка документів (шматків тексту)
docs = [
    Document(
        page_content=f"passage: {chunk}", 
    )

    for chunk in texts
]

print(f"Успішно створено {len(docs)} чанків.")

# 4. Створення колекції та збереження векторів у Qdrant
# vector_store = QdrantVectorStore.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     url=qdrant_url,
#     collection_name="ukrhistory11_2",
# )


vector_store = QdrantVectorStore.from_documents(
    docs,
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    collection_name="ukrhistory11_sparse",
    url=qdrant_url,
)



# 5. Семантичний пошук схожих документів
# Для E5 обов'язково додаємо префікс "query: " до запиту
# query = "query: Розкажи про базу на Rust"
# found_docs = vector_store.similarity_search(query, k=1)

# print("\nЗнайдений релевантний шматок тексту:")
# # Прибираємо префікс "passage: " при виводі, якщо потрібно
# print(found_docs[0].page_content.replace("passage: ", "", 1))