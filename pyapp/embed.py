import configparser
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

print(f"STEP  #1.1 Connect to the Milvus server")

# connections.connect("default", host="standalone", port="19530")
cfp = configparser.RawConfigParser()
cfp.read('config.ini')
milvus_uri = cfp.get('example', 'uri')
token = cfp.get('example', 'token')

connections.connect("default",
                    uri=milvus_uri,
                    token=token)
print(f"STEP  #1.2 Define the collection schema")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="ristodata", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, description="demo for inserting BERT vectors")
print(schema)
print(f"STEP  #1.3 Specify the collection name")
collection_name = "ristobot"

print(f"STEP  #1.4 Drop the collection if it exists")

if utility.has_collection(collection_name):
    collection = Collection(collection_name)
    collection.drop()

print(f"STEP  #1.5 Create the collection")

collection = Collection(collection_name, schema=schema)

print(f"STEP  #1.6 Load the SentenceTransformer model")

model = SentenceTransformer('/pyapp/models/multi-qa-MiniLM-L6-cos-v1/')

sentences = [
    "This is an example sentence",
    "Each string here gets embedded",
    "The embeddings can then be indexed using Milvus"
]

print(f"STEP  #1.7 Generate sentence embeddings using the SentenceTransformer model")

sentence_embeddings = model.encode(sentences)
print(sentence_embeddings)

print(f"STEP  #1.8 prepare data")

for vector in sentence_embeddings:
    print(vector)

entities = [
    {"ristodata": vector.tolist()} for vector in sentence_embeddings
]

print(entities)


print(f"STEP  #1.9 insert")

insert_result = collection.insert(entities, collection_name)
print(insert_result)

print(f"STEP  #1.10 Flush Collection")
collection.flush()


print(f"STEP  #1.11 disconnect")
connections.disconnect("default")