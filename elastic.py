from elasticsearch import Elasticsearch, AsyncElasticsearch

host = 'http://localhost:9200'
es = Elasticsearch(hosts=host, max_retries=10, retry_on_timeout=True, verify_certs=False, ssl_show_warn=False)
es_async = AsyncElasticsearch(hosts=host, max_retries=10, retry_on_timeout=True, verify_certs=False, ssl_show_warn=False)
