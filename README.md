
# 🇺🇦 RAG HistoryAgent

<p align="center">  
<strong>An AI-powered assistant for exploring Ukrainian history.</strong>  
<br />  
Retrieval-Augmented Generation · Hybrid Search · Reranking  
</p>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![aiogram](https://img.shields.io/badge/aiogram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)
![BM25](https://img.shields.io/badge/BM25-Sparse%20Retrieval-6A5ACD?style=for-the-badge)
![multilingual--e5--large](https://img.shields.io/badge/Embeddings-multilingual--e5--large-8A2BE2?style=for-the-badge)
----------

## Overview

**RAG HistoryAgent** is an AI assistant designed to help students explore Ukrainian history through natural-language questions.

The system retrieves relevant historical passages from a knowledge base and uses a large language model (LLM) to generate contextual answers grounded in the retrieved information.

Rather than relying solely on the language model's internal knowledge, the project combines semantic retrieval, keyword-based search, hybrid ranking, and reranking to improve the relevance of the context supplied to the model.

## Demo

<p align="center">
  <img src="assets/demo.png" alt="RAG HistoryAgent demonstration" width="850" />
</p>

<details>
<summary>🌐 See in English</summary>

**Request:** Hi, tell me who you are, then tell me about Khrushchev.

**Response:**

Hello! I'm a historical assistant that helps you explore history. Ask me about events, dates, historical figures, causes, and consequences.

Now, let's talk about **Nikita Khrushchev (1894–1971)**.

### Early life and ties to Ukraine

Khrushchev worked in the Donbas mines and learned to operate mining equipment at the Donetsk Geological Technical School. In 1931, he began his political career in Moscow, and in 1935 became head of the Moscow City and Regional Party Committees.

### De-Stalinization

His report *On the Cult of Personality and Its Consequences* initiated de-Stalinization — the criticism of Stalin's cult of personality. This process divided Soviet society and challenged the existing totalitarian system. Khrushchev argued that the party could not return to Stalinism.

### Removal from power

In October 1964, Khrushchev was removed from power by party officials led by Leonid Brezhnev. The official reason cited his age and deteriorating health. His removal marked the beginning of the Brezhnev era and the period later known as the *Era of Stagnation*.

</details>

<p align="center">
  <em>Exploring Ukrainian history through natural-language questions.</em>
</p>

## Key Features

-   **Retrieval-Augmented Generation** — generates answers using relevant historical context retrieved from the knowledge base.
    
-   **Multilingual Semantic Search** — uses `multilingual-e5-large` to represent documents and queries as dense embeddings.
    
-   **Hybrid Retrieval** — combines dense vector search with BM25 keyword retrieval to capture both semantic meaning and exact terminology.
    
-   **Reranking** — refines the initial retrieval results to prioritize passages most relevant to the user's question.
    
-   **Vector Database** — uses Qdrant to store and retrieve document embeddings.
    
-   **Local LLM Inference** — integrates with LM Studio to run supported language models locally.
    
-   **Educational Focus** — provides a conversational interface for exploring Ukrainian history.
    

## Architecture

```mermaid

flowchart TD
    A["User Question"] --> B["Intent Classification<br/>GLM-5.3-Flash"]

    B --> C{"Query Intent"}

    C -->|General| D["General Response Generation<br/>GLM-5.3-Flash"]
    D --> J["Telegram Bot Response"]

    C -->|Historical| E["Query Preprocessing<br/>GLM-5.3-Flash"]

    E --> F["Dense Vector Search<br/>multilingual-e5-large"]
    E --> G["Sparse Keyword Search<br/>BM25"]

    F --> H["Qdrant Hybrid Retrieval<br/>Top 30 Documents"]
    G --> H

    H --> I["Cross-Encoder Reranking<br/>bge-reranker-v2-m3"]
    I --> K["Top 5 Relevant Passages"]

    K --> L["Answer Generation<br/>GLM-5.3-Flash"]
    L --> M["Answer Verification<br/>LLM Review"]
    M --> J

    style B fill:#1e3a8a,color:#ffffff,stroke:#60a5fa
    style C fill:#713f12,color:#ffffff,stroke:#facc15
    style H fill:#164e63,color:#ffffff,stroke:#06b6d4
    style I fill:#4c1d95,color:#ffffff,stroke:#a78bfa
    style M fill:#14532d,color:#ffffff,stroke:#4ade80
    style D fill:#164e63,color:#ffffff,stroke:#06b6d4

```

The pipeline combines dense semantic search and sparse BM25 retrieval using Qdrant's hybrid search. A cross-encoder reranks the retrieved candidates, selecting the five most relevant passages for answer generation. The generated answer is then reviewed by a second LLM call to identify and correct claims unsupported by the retrieved context before being returned through Telegram.
## Tech Stack

 - Python    
 - LangChain      
 - Embeddings
			            `multilingual-e5-large`
  
 - Qdrant      
 - BM25
 - aiogram

## Getting Started

### Prerequisites

-   Python
    
-   [uv](https://docs.astral.sh/uv/)
    
-   A running Qdrant instance with the required collection and indexed documents
    
-   [LM Studio](https://lmstudio.ai/) with a compatible model loaded and its local API enabled
    

### Installation

Clone the repository:

```
git clone https://github.com/water980/histagent
cd histagent
```

Install the project dependencies:

```
uv sync
```

### Configuration

Create a `.env` file and configure the environment variables required by the application.

Keep API keys and other secrets out of version control.

### Run the Application


```
chmod +x start_qdrant.sh 
./start_qdrant.sh # start qdrant using docker
chmod +x prepare_book.sh
./prepare_book.sh  # preparing book using scripts
uv run python main.py # run the application
```

Adjust the command to match your actual project structure.

## Project Goals

RAG HistoryAgent explores how modern retrieval techniques can improve the usefulness of LLM-based educational applications, particularly when answering domain-specific questions in Ukrainian.

The project brings together semantic search, lexical retrieval, reranking, vector storage, and local language model inference in a single application.
