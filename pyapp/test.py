import configparser
import time
import random
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from sentence_transformers import SentenceTransformer

bert_model_dir='/pyapp/models/multi-qa-MiniLM-L6-cos-v1/'
collection_name = "book"

if __name__ == '__main__':
    # connect to milvus
    cfp = configparser.RawConfigParser()
    cfp.read('config.ini')
    milvus_uri = cfp.get('milvus_ristobot', 'uri')
    token = cfp.get('milvus_ristobot', 'token')

    connections.connect("default",
                        uri=milvus_uri,
                        token=token)
    print(f"Connecting to DB: {milvus_uri}")

    # Check if the collection exists
    check_collection = utility.has_collection(collection_name)
    if check_collection:
        drop_result = utility.drop_collection(collection_name)
    print("Success!")

    # create a collection with customized primary field: book_id_field
    dim = 384
    book_id_field = FieldSchema(name="book_id", dtype=DataType.INT64, is_primary=True, description="customized primary id")
    word_count_field = FieldSchema(name="word_count", dtype=DataType.INT64, description="word count")
    book_intro_field = FieldSchema(name="book_intro", dtype=DataType.FLOAT_VECTOR, dim=dim)
    sentence_field = FieldSchema(name="sentence", dtype=DataType.VARCHAR,description="book sentence",  max_length=1000,default_value="")
    schema = CollectionSchema(fields=[book_id_field, word_count_field, book_intro_field, sentence_field],
                          auto_id=False,
                          description="my first book collection")
    print(f"Creating ristobot collection: {collection_name}")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Schema: {schema}")
    print("Success!")

    # insert data with customized ids
    nb = 3
    insert_rounds = 2
    start = 0           # first primary key id
    total_rt = 0        # total response time for inert
    print(f"Inserting {nb * insert_rounds} entities... ")

    print(f"STEP   Generate sentence embeddings using the SentenceTransformer model")

    model = SentenceTransformer(bert_model_dir)



    # during insertion
    sentences = [
        "mela",
        "pera",
        "banana"
    ]
    sentence_embeddings = model.encode(sentences)
    book_ids = list(range(nb))
    id_sentence_map = dict(zip(book_ids, sentences))




    #     for i in range(insert_rounds):
    #         book_ids = [i for i in range(start, start+nb)]
    #         word_counts = [random.randint(1, 100) for i in range(nb)]
    #         #book_intros = [[random.random() for _ in range(dim)] for _ in range(nb)]
    #         book_intros = sentence_embeddings[i].tolist()
    #         entities = [book_ids, word_counts, book_intros]
    #         t0 = time.time()
    #         ins_resp = collection.insert(entities)
    #         ins_rt = time.time() - t0
    #         start += nb
    #         total_rt += ins_rt
    print(f"STEP  #1.8 prepare data")
    book_ids = [i for i in range(nb)]
    word_counts = [random.randint(1, 100) for i in range(nb)]
    book_intros = [embedding.tolist() for embedding in sentence_embeddings]
    #sentence = [sentence for sentence in sentences]
    entities = [book_ids, word_counts, book_intros,sentences]

    t0 = time.time()
    ins_resp = collection.insert(entities)
    t1 = time.time()
    print(f"Succeed in {round(t1-t0, 4)} seconds!")
    # print(f"collection {collection_name} entities: {collection.num_entities}")

    # flush
    print("Flushing...")
    start_flush = time.time()
    collection.flush()
    end_flush = time.time()
    print(f"Succeed in {round(end_flush - start_flush, 4)} seconds!")
    # build index
    index_params = {"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}}
    t0 = time.time()
    print("Building AutoIndex...")
    collection.create_index(field_name=book_intro_field.name, index_params=index_params)
    t1 = time.time()
    print(f"Succeed in {round(t1-t0, 4)} seconds!")

    # load collection
    t0 = time.time()
    print("Loading collection...")
    collection.load()
    t1 = time.time()
    print(f"Succeed in {round(t1-t0, 4)} seconds!")

    # search

    search_str = "pera"


    nq = 1
    search_params = {"metric_type": "L2",  "params": {"level": 2}}
    topk = 1

    mela_embedding = model.encode([search_str])[0].tolist()    # encode the text "mela"
    search_vec = [mela_embedding]                          # use this embedding as the search vector


    print(f"Searching for '{search_str}'...")
    t0 = time.time()
    results = collection.search(search_vec,
                            anns_field=book_intro_field.name,
                            param=search_params,
                            limit=topk,
                            guarantee_timestamp=1)
    t1 = time.time()
    print(f"Result:{results}")
    print(f"search latency: {round(t1-t0, 4)} seconds!")

    matched_book_id = 0
    # after searching
    for hit in results[0]:
        print('hit id: ', hit)
        print('hit id: ', hit.id)
        print('hit distance: ', hit.distance)

        # after searching
        matched_book_id = hit.id
        matched_sentence = id_sentence_map[matched_book_id]
        print(f"Matched Sentence: {matched_sentence}")



    # Conduct a query
    given_id = matched_book_id   # Replace this with your given id
    res = collection.query(
      expr = f"book_id == {given_id}",
      output_fields = ["book_id", "word_count","sentence"]
    )

    # Check the query result
    for item in res:
        print(f"book_id: {item['book_id']},  word_count: {item['word_count']} ,  sentence: {item['sentence']}")
        connections.disconnect("default")

