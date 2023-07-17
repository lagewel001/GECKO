import faiss
import json
import itertools
import operator
import numpy as np
import pandas as pd
from elasticsearch.helpers import bulk
from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

import paths_config
from elastic import es
from global_functions import secure_request
from model.roberta.MLM_trainer import extend_tokenizer
from pipeline.logical_forms import uri_to_code

GRAPH_DB_HOST = '<host>:7200'
REPOSITORY = paths_config.GRAPH_DB_REPO
sparql = SPARQLWrapper(f"http://{GRAPH_DB_HOST}/repositories/{REPOSITORY}")
sparql.setCredentials('<username>', '<password>')
sparql.setReturnFormat(JSON)

# Used for creating index map of node codes in Faiss
tokenizer = AutoTokenizer.from_pretrained(paths_config.TOKENIZER_PATH)
model = SentenceTransformer(paths_config.SENT_TRANSFORMER_PATH)


def index_nodes():
    tables = secure_request('https://odata4.cbs.nl/CBS/datasets', json=True, max_retries=9, timeout=20)['value']
    questions = pd.read_csv(paths_config.DATASET_PATH, delimiter=';').sexp
    question_tables = questions.str.extract(r'(?<=\(VALUE \()(.*?)(?=\s)', expand=False).unique()

    kerncijfers = [t['Identifier'] for t in tables
                   if ('kerncijfers' in t['Title'].lower() and t['Status'] == 'Regulier')
                   or t['Identifier'] in question_tables]

    tables = {}
    nodes = {}
    for table in tqdm(kerncijfers, desc="Getting nodes", bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
        props = secure_request(f"https://odata4.cbs.nl/CBS/{table}/Properties", json=True, max_retries=9, timeout=20)
        tables[f"https://opendata.cbs.nl/#/CBS/nl/dataset/{table}"] = {
            'body': ' '.join([props['Title'], props['Description'], props['Summary'], props['LongDescription']]),
            'type': 'table'
        }

        # Get all measures and dimensions for a table. Skip the time and geo dimensions
        query = (f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX qb: <http://purl.org/linked-data/cube#>
            
            SELECT ?o ?label WHERE {{ 
                ?s qb:measure|qb:dimension ?o .
                FILTER NOT EXISTS {{
                    VALUES ?dim {{'TimeDimension' 'GeoDimension'}}
                    ?o ?has_type ?dim .
                }}
                ?s dct:identifier ?id .
                OPTIONAL {{ ?o skos:prefLabel|skos:altLabel|skos:definition|dct:description|dct:subject ?label }}
                FILTER (?id = "{table}")
                FILTER (!BOUND(?label) || lang(?label) = "nl")
            }}
        """)
        sparql.setQuery(query)
        sparql.method = 'GET'
        try:
            result = sparql.queryAndConvert()['results']['bindings']
            props = [(r['o']['value'], (r.get('label', False) or {'value': ''})['value']) for r in result]
            it = itertools.groupby(props, operator.itemgetter(0))
            for key, subiter in it:
                val = ' '.join(prop[1] for prop in subiter)
                nodes[key] = {'body': val, 'type': 'node'}
                tables[f"https://opendata.cbs.nl/#/CBS/nl/dataset/{table}"]['body'] += ' ' + val
        except Exception as e:
            print(f"Failed to fetch table nodes: {e}")

    params = {
        "k1": 1.2,  # positive tuning parameter that calibrates the document term frequency scaling
        "b": 0.3,   # 0.75,  # parameter that determines the scaling by document length
        "d": 1.0    # makes sure the component of term frequency normalization by doc. length is properly lower-bounded
    }

    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "index": {
            "similarity": {
                "bm25_plus": {
                    "type": "scripted",
                    "script": {
                        "source": f"double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0;"
                                  f"return query.boost * idf * (("
                                  f"(doc.freq * ({params['k1']} + 1))/(doc.freq + ({params['k1']} * (1 - {params['b']} + "
                                  f"({params['b']} * doc.length/(field.sumTotalTermFreq/field.docCount)))))) + {params['d']});"
                    }
                }
            },
            "store.preload": ["nvd", "dvd"],
        },
        "analysis": {
            "filter": {
                "dutch_stop": {
                    "type": "stop",
                    "ignore_case": True,
                    "stopwords": ["_dutch_", "hoeveel", "waar", "waarom", "aantal", "welke", "wanneer", "waardoor", "gemiddeld"]
                },
                "dutch_stemmer": {
                    "type": "stemmer",
                    "language": "dutch"
                },
                "index_shingle": {
                    "type": "shingle",
                    "min_shingle_size": 2,
                    "max_shingle_size": 3,
                },
                "ascii_folding": {
                    "type": "asciifolding",
                    "preserve_original": False
                },
            },
            "analyzer": {
                "graph-nl": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "dutch_stop",
                        "apostrophe",
                        "ascii_folding",
                        "dutch_stemmer",
                        "index_shingle",
                    ]
                }
            }
        }
    }

    mappings = {
        'properties': {
            'unique_id': {
                'type': 'keyword'
            },
            'body': {
                'type': 'text',
                "term_vector": "with_positions_offsets",
                'analyzer': "graph-nl",
                'search_analyzer': "graph-nl",
                "similarity": "bm25_plus",
            },
            'type': {
                'type': 'keyword'
            },
            "embedding_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            }
        }
    }

    if es.indices.exists(index='graph-nl'):
        es.indices.delete(index='graph-nl')
    es.indices.create(index='graph-nl', settings=settings, mappings=mappings)

    node_embeddings = {}
    ops = []
    try:
        for i, (id_, d) in enumerate(tqdm((nodes | tables).items(), desc=f"Indexing nodes",
                                          bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
            embedding = model.encode(d['body'])
            node_embeddings[id_] = embedding

            search_doc = {
                'unique_id': id_,
                'body': d['body'],
                'type': d['type'],
                'embedding_vector': embedding
            }
            ops.append({
                '_index': 'graph-nl',
                '_id': id_,
                '_source': search_doc
            })
            if i % 50 == 0 and i > 0:
                bulk(es, ops, chunk_size=50, request_timeout=30)
                ops = []
        if len(ops) > 0:
            bulk(es, ops, chunk_size=50, request_timeout=30)
            ops = []

        embedding_arr = np.array(list(node_embeddings.values()))
        faiss_index = faiss.IndexFlatIP(embedding_arr.shape[1])
        index_map = faiss.IndexIDMap2(faiss_index)

        # Extend codes of tokenizer
        # TODO: codes that are removed from OData4/the graph will still be kept in the tokenizer vocab
        global tokenizer
        uri_codes = [uri_to_code(k) for k in node_embeddings.keys()]
        tokenizer = extend_tokenizer(tokenizer=tokenizer,
                                     save_path=paths_config.TOKENIZER_EXT_PATH,
                                     extra_tokens=np.array(uri_codes))

        vocab_id_map = np.array(tokenizer.convert_tokens_to_ids(uri_codes))
        mask = np.where(vocab_id_map != tokenizer.unk_token_id)[0]  # don't store embeddings for codes not in the vocab
        if len(mask) != len(vocab_id_map):
            print("Warning: for some reason the size of the Faiss index and tokenizer vocab is not equal!")
        index_map.add_with_ids(embedding_arr[mask], vocab_id_map[mask])

        faiss.write_index(index_map, paths_config.FAISS_PATH)

        uri_map = {int(vocab_id): uri for vocab_id, uri in
                   zip(vocab_id_map[mask], np.array(list(node_embeddings.keys()))[mask])}
        with open(paths_config.FAISS_URI_MAP_PATH, 'w') as file:
            file.write(json.dumps(uri_map, indent=4))

        # For debugging purposes only
        k = 25
        xq = model.encode(["Hoeveel mensen gingen er met de trein op vakantie naar het buitenland?"])
        knn = index_map.search(xq, k)[1][0]

        node_descriptions = {uri_to_code(k): v for k, v in (nodes | tables).items()}
        for code in tokenizer.convert_ids_to_tokens(knn):
            print(node_descriptions[code])
    except StopIteration:
        print('StopIteration')
        bulk(es, ops, chunk_size=50, request_timeout=30)
    except Exception as e:
        print('Indexing stopped unexpectedly')
        print(e)


if __name__ == "__main__":
    index_nodes()
