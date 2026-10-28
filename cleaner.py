import re

from pypdf import PdfReader

PDF = "histbook.pdf"
OUTPUT = "histbook_clean.txt"

TASK_START = [
    "Проаналізуйте",
    "Які твердження",
    "Які особливості",
    "Наведіть факти",
    "Прокоментуйте",
    "Підготуйте",
    "Пригадайте",
    "Подискутуйте",
    "Порівняйте",
    "Поясніть",
    "Схарактеризуйте",
    "Визначте",
    "Поміркуйте",
    "Обговоріть",
    "Опрацюйте",
    "Виконайте",
]

TASK_HEADERS = [
    "ЗАПИТАННЯ І ЗАВДАННЯ",
    "ЗАПИТАННЯ ТА ЗАВДАННЯ",
    "ПЕРЕВІРТЕ, ЧОГО НАВЧИЛИСЯ",
    "ЯК ПРОВЕСТИ САМООЦІНЮВАННЯ",
]


def is_task_start(line):
    line = line.strip()

    if any(line.startswith(x) for x in TASK_START):
        return True

    if any(x in line.upper() for x in TASK_HEADERS):
        return True

    return False


def is_new_section(line):
    line = line.strip()

    # § 1, § 2...
    if re.match(r"^§\s*\d+", line):
        return True

    return False


reader = PdfReader(PDF)

result = []

for page_num, page in enumerate(reader.pages, 1):
    print(f"Сторінка {page_num}/{len(reader.pages)}")

    text = page.extract_text()

    if not text:
        continue

    lines = text.splitlines()

    skip = False

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Початок завдання
        if is_task_start(line):
            skip = True
            continue

        # Повертаємося до тексту на новому параграфі
        if skip and is_new_section(line):
            skip = False

        if not skip:
            result.append(line)


text = "\n".join(result)

# Прибираємо зайві пробіли
text = re.sub(r"[ \t]+", " ", text)

# Прибираємо багато порожніх рядків
text = re.sub(r"\n{3,}", "\n\n", text)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nГотово: {OUTPUT}")

