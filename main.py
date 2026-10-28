import asyncio
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from sentence_transformers import CrossEncoder

load_dotenv()

# # 1. Ініціалізуємо ту саму модель ембеддінгів, якою кодували текст
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device': 'cuda'},  # або 'cpu'
    encode_kwargs={'normalize_embeddings': True}
)


reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

# 2. Підключаємося до існуючої колекції
# vector_store = QdrantVectorStore.from_existing_collection(
#     embedding=embeddings,
#     collection_name="ukrhistory11_2",
#     url="http://localhost:6333",
# )


sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="ukrhistory11_sparse",
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
)

# 3. Тепер можна одразу шукати
tg_token = os.getenv("TG_TOKEN")
llm_token = os.getenv("LLM")

# llm = ChatOpenAI(
#     model="ministral-3-8b-instruct-2512", # точний ідентифікатор з вашого терміналу
#     base_url="http://localhost:1234/v1",  # локальний порт LM Studio
#     api_key=llm_token,                  # заглушка, оскільки ключ не потрібен
#     temperature=0,
#     max_tokens=1024,
# )

llm = ChatOpenAI(
    model="qwen3-14b", # точний ідентифікатор з вашого терміналу
    base_url="http://localhost:1234/v1",  # локальний порт LM Studio
    api_key=llm_token,                  # заглушка, оскільки ключ не потрібен
    temperature=0,
    max_tokens=2048,
)


def preprocess_query(query: str) -> str:
    prompt = f"""
Ти перетворюєш запит для пошуку в українському історичному підручнику.

Виділи найважливіші імена людей, назви організацій, подій, установ,
країн та інших сутностей.

Використовуй нормальну словникову форму імен та назв.
Не додавай пояснень.
Не змінюй написання власних назв на інше слово.

Приклади:

Query: Хто такий Дмитро Мануїльський?
Result: Дмитро Мануїльський

Query: Розкажи про Мануїльского
Result: Мануїльський

Query: Розкажи про Сан-францизьку конференцію і наркомат Мануїльського
Result: Сан-Франциська конференція, наркомат Мануїльський

Query: Розкажи про Хрущова
Result: Хрущов

Query: Що відомо про Сталіна?
Result: Сталін

Query: {query}
Result:
"""

    ai_message = llm.invoke(prompt)
    return ai_message.content.strip()

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


def search(query):
    search_query = f"query: {preprocess_query(query)}"
    
    print(f"search_query: {search_query}")
    
        # 1. Hybrid retrieval: беремо більше кандидатів
    found_docs = vector_store.similarity_search(
        search_query,
        k=30,
    )
    
        # 2. Reranking
    reranked_docs = rerank(
        query=query,
        documents=found_docs,
        limit=5,
    )
    return reranked_docs



def request(query: str) -> str:
    reranked_docs = search(query)
    # Збираємо контекст, очищаючи префікси passage
    context = ""

    for i, doc in enumerate(reranked_docs, 1):
        text = doc[1].page_content.replace("passage: ", "", 1)
        context += f"\n--- Матеріал {i} ---\n{text}\n"


    prompt = f"""
    Ти — історичний асистент для учня.

    Твоє головне правило: відповідай ТІЛЬКИ на основі наданих матеріалів.



    Відповідай як звичайний історичний асистент для учня.

    Якщо питання потребує пояснення — пояснюй.
    Якщо можна відповісти коротко — відповідай коротко.
    Не перераховуй усі знайдені факти. Використовуй лише ті, які безпосередньо допомагають відповісти на питання.

    Форматування:

    * використовуй HTML для Telegram;
    * <b> для важливих понять;
    * <i> для термінів або назв;
    * <u> використовуй рідко;
    * не використовуй Markdown;
    * не використовуй ``` у відповіді.

    Перед формуванням відповіді подумки перевір:

    1. Які факти безпосередньо відповідають на питання?
    2. Чи збігаються дати та події?
    3. Чи не взяв я факт з іншого історичного періоду?
    4. Чи справді матеріали підтверджують мій причинно-наслідковий висновок?
    5. Чи немає в моїй відповіді інформації, якої немає в матеріалах?

    МАТЕРІАЛИ:
    {context}

    ЗАПИТАННЯ:
    {query}

    ВІДПОВІДЬ:
    """


    print(context + "\n\n\n")
    # Викликаємо модель та обов'язково дістаємо .content
    ai_message = llm.invoke(prompt)
    return ai_message.content, reranked_docs



def check_request(query: str, res: str, reranked_docs) -> str:
    # search_query = f"query: {query}"
    # found_docs = vector_store.similarity_search(search_query, k=7)

    context = ""
    for i, doc in enumerate(reranked_docs, 1):
        text = doc[1].page_content.replace("passage: ", "", 1)
        context += f"\n--- Матеріал {i} ---\n{text}\n"

    prompt = f"""
Ти — редактор історичного асистента.

Твоє завдання — перевірити готову відповідь на запитання
та ВИПРАВИТИ її так, щоб вона не містила тверджень,
яких немає в наданих матеріалах.

ВАЖЛИВО:

1. Використовуй ТІЛЬКИ інформацію з матеріалів.

2. Не додавай жодних власних історичних знань.

3. Перевіряй кожен конкретний факт:
   - дати;
   - імена;
   - події;
   - числа;
   - причини;
   - наслідки.

4. Якщо твердження відповіді НЕ підтверджується матеріалами —
   видали його або заміни на формулювання, яке підтверджується.

5. Особливо уважно перевіряй причинно-наслідкові зв'язки.

   Наприклад, якщо матеріал каже:
   «під час неврожаю збільшення частки кукурудзи призводило
   до проблем із кормом і хлібом»,

   НЕ можна писати:
   «кукурудзяна політика спричинила голод»,

   якщо такого причинного зв'язку прямо немає.

6. Не переносити факти між різними історичними періодами.

7. Не використовуй інформацію з матеріалу лише тому,
   що вона тематично схожа на питання.

8. Якщо у відповіді є правильна інформація — залиш її.

9. Не потрібно переписувати відповідь без причини.
   Змінюй тільки те, що є неправильним, неточним,
   непідтвердженим або недоречним.

10. Не пояснюй, що саме ти виправив.
    Поверни ТІЛЬКИ готову виправлену відповідь.

11. Не згадуй слова «матеріал», «контекст», «джерело»,
    «перевірка» або «RAG».

12. Не додавай нових фактів для покращення відповіді.

ФОРМАТУВАННЯ:

- HTML для Telegram.
- <b> для важливих понять.
- <i> для термінів або назв.
- Не використовуй Markdown.
- Не використовуй ```.

МАТЕРІАЛИ:

{context}

ЗАПИТАННЯ:

{query}

ПОПЕРЕДНЯ ВІДПОВІДЬ:

{res}

ВИПРАВЛЕНА ВІДПОВІДЬ:
"""

    ai_message = llm.invoke(prompt)

    return ai_message.content


bot = Bot(
        tg_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
dp = Dispatcher()



async def main():
    await dp.start_polling(bot)

@dp.message()
async def massage(message):
   res, docs = request(message.text)
   checked_res = check_request(message.text, res, docs)
   await message.answer(checked_res) 

asyncio.run(main())
















