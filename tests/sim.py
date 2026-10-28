from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embeddings = SentenceTransformer("intfloat/multilingual-e5-large")

correct_chunk = """великого українця Богдана Г аврилишина. Виберіть кілька фактів із його 
 біографії, які вас вразили. Проаналізуйте висловлювання Б. Г аврилишина 
про обов’язки людини. Поміркуйте, чи погоджуєтеся з усіма. Сформулюйте 
за таким самим принципом власні обов’язки до сім’ї, батьків до дітей, до суспільства 
тощо. Представте вашу роботу в класі.
«Обов’язки людини:
 до сім’ї:
 бережи культурну спадщину своїх предків;
 стався до батьків із любов’ю та повагою, за потреби допомагай їм;"""

query = "query: Мануїльський"

query_vector = embeddings.encode(query, normalize_embeddings=True)
chunk_vector = embeddings.encode(
    "passage: " + correct_chunk,
    normalize_embeddings=True
)
score = cosine_similarity(
    [query_vector],
    [chunk_vector]
)[0][0]

print("Similarity:", score)