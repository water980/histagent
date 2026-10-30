from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import llm


general_prompt = ChatPromptTemplate.from_template("""
Ти — український асистент для школярів.

Відповідай природно, доброзичливо та зрозуміло.
Можеш спілкуватися на будь-які теми.
Не вигадуй інформацію про себе.

Запитання:
{query}

Відповідь:
""")

general_chain = (
    general_prompt
    | llm
    | StrOutputParser()
)
