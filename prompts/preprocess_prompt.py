from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import llm


preprocess_prompt = ChatPromptTemplate.from_template("""
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
""")

preprocess_chain = (
    preprocess_prompt
    | llm
    | StrOutputParser()
)
