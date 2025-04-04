from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json

filePath = "scrape_prac/outputFiles/lkdScr2.json"
# es = Elasticsearch("http://localhost:9200")
es = Elasticsearch(hosts=[{"host": "localhost", "port": 9200, "scheme": "http"}])
model = SentenceTransformer("all-MiniLM-L6-v2")

with open(filePath, "r") as file:
    data = json.load(file)

embeddings = []
indexName = "embed_prac_index"
embedding_dim = 384


if not es.indices.exists(index=indexName):
    es.indices.create(
        index = indexName,
        body={
            "mappings":{
                "properties":{
                    "embededText":{"type" : "text"},
                    "embedding":{"type":"dense_vector", "dims":embedding_dim},
                    "raw_json":{"type":"object", "enabled":True}
                }
            }
        }
    )



for i, job in enumerate(data):
    
    embededText = json.dumps(job, indent=2)
    embedding = model.encode(embededText).tolist()

    doc = {
        "embededText" : embededText,
        "embedding" : embedding,
        "job" : job
    }

    # es.index(index = indexName, body = doc)
    es.index(index = indexName, document= doc)
    print(f"✅ Inserted job {i + 1}/{len(data)}")


# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
# query = "I want a job in Generative AI with LLMs and MLOps"
# query_vector = model.encode(query).tolist()
# print(query_vector)