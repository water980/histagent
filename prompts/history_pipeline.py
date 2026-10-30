from models import (
    reranker,
    vector_store
)
from prompts.answer_prompt import answer_chain
from prompts.preprocess_prompt import preprocess_chain
from prompts.verification_prompt import verification_chain
from config import TG_TOKEN
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

def rerank(query, documents, limit=5):
    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda item: item[0],
        reverse=True,
    )

    return ranked[:limit]


def retrieve_context(inputs: dict) -> dict:
    query = inputs["query"]

    processed_query = preprocess_chain.invoke({
        "query": query
    })

    search_query = f"query: {processed_query}"

    print(f"search_query: {search_query}")

    found_docs = vector_store.similarity_search(
        search_query,
        k=30,
    )

    reranked_docs = rerank(
        query=query,
        documents=found_docs,
        limit=5,
    )

    context_parts = []

    for i, (score, doc) in enumerate(reranked_docs, 1):
        text = doc.page_content.replace("passage: ", "", 1)
        context_parts.append(
            f"\n--- Матеріал {i} ---\n{text}\n"
        )

    context = "".join(context_parts)

    print(context)

    return {
        "query": query,
        "context": context,
        "reranked_docs": reranked_docs,
    }


history_pipeline = (
    RunnableLambda(retrieve_context)
    | RunnablePassthrough.assign(
        answer=answer_chain
    )
    | RunnablePassthrough.assign(
        checked_answer=verification_chain
    )
)
