import faiss
import json
import logging
import numpy as np
import re
import string
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from rdflib import Graph, QB, URIRef
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from typing import Tuple, Dict

import paths_config
from elastic import es
from pipeline.logical_forms import uri_to_code, TABLE, MSR, DIM
from pipeline.sparql_controller import explode_subgraph, explode_subgraph_msr_dims_only
from pipeline.query_expander_controller import expand_query

logger = logging.getLogger(__name__)


class EntityRecognizer:
    encoder = SentenceTransformer(paths_config.SENT_TRANSFORMER_PATH)
    faiss_index = faiss.read_index(paths_config.FAISS_PATH)
    tokenizer = AutoTokenizer.from_pretrained(paths_config.TOKENIZER_EXT_PATH)  # used for index mapping

    with open(paths_config.FAISS_URI_MAP_PATH, 'r') as file:
        uri_map = json.load(file)
        inv_uri_map = {v: k for k, v in uri_map.items()}

    @staticmethod
    def get_candidate_nodes_bm25(query, k=25, full_graph: bool = False, verbose=False) -> \
            Tuple[Graph, Dict, Dict, Dict]:
        """
            Find all candidate entities corresponding with the query in the graph using the Elastic index. The result
            will be the candidate measures and dimensions with their corresponding matching scores, and the resulting
            subgraph of all the connected tables and their connected measures and dimensions from these candidates.

            :param query: NL search query
            :param k: number of closest ES-nodes to consider
            :param full_graph: fetch full graph, including dimension hierarchy relations (slower)
            :param verbose: print the executed SPARQL query
            :returns: (Subgraph, {candidate measures: scores}, {candidate dims: scores})
        """
        query = [w for w in word_tokenize(query) if w not in string.punctuation]
        expanded_query = expand_query(query)

        # TODO: get top k-nodes, explode and ask ES separately for all BM25 codes per node in the subgraph
        q = {
            "track_total_hits": True,
            "_source": ["_id"],
            "min_score": 0.01,
            "size": k,
            "query": {
                "match": {
                    "body": {
                        "minimum_should_match": 1,
                        "query": ' '.join(expanded_query)
                    }
                }
            }
        }

        if verbose:
            logger.debug("Querying Elastic...")

        resp = es.search(body=q, index='graph-nl')
        er_nodes = [n['_id'] for n in resp.body['hits']['hits']]
        if len(er_nodes) == 0:
            raise ValueError("No nodes found in Elastic that correspond with the given query.")

        # Get all (table, measure/dimension, entity) combinations following from the relations of the candidate ER nodes
        if verbose:
            logger.debug("Exploding nodes and creating subgraph...")

        # EXPLODE!
        expl_fun = explode_subgraph_msr_dims_only if not full_graph else explode_subgraph
        g = expl_fun(er_nodes, verbose=verbose)

        # Get BM25-scores of all nodes in subgraph
        subgraph_nodes = [str(s) for s in set(g.subjects()) | set(g.objects())]
        q = {
            "size": len(subgraph_nodes),
            "_source": ["_id"],
            "query": {
                "bool": {
                    "should": [
                        {"match": {"body": ' '.join(expanded_query)}}
                    ],
                    "filter": [
                        {"terms": {"_id": subgraph_nodes}}
                    ]
                }
            },
            "highlight": {  # TODO: phrase recognized entities and only return sorted phrase highlights
                "pre_tags": ['<match>'],
                "post_tags": ['</match>'],
                "order": "score",
                "number_of_fragments": 1,   # TODO: for phrase highlighting increase fragment size to something sensible
                "highlight_query": {
                    "bool": {
                        "minimum_should_match": 0,
                        "should": [{
                            "multi_match": {
                                "query": ' '.join(expanded_query),
                                "type": "best_fields",
                                "fields": ['body'],
                            }
                        }]
                    }
                },
                "fields": {
                    "body": {
                        "boundary_scanner_locale": "NL-nl",
                        "type": "fvh"
                    }
                }
            }
        }

        er_nodes = {}
        resp = es.search(body=q, index='graph-nl')
        for doc in resp.body['hits']['hits']:
            er_nodes[doc['_id']] = {
                "score": doc['_score'],
                "matched_words": (set([s.lower() for s in re.findall(r"<match>(.*?)</match>",
                                                                     doc['highlight']['body'][0])])
                                  if 'highlight' in doc else None)
            }

        tables = set(g.subjects(QB.measure, None))
        measures = set(g.objects(None, QB.measure))
        dims = set(g.objects(None, QB.dimension))

        return (g,
                {uri_to_code(t): er_nodes[str(t)] for t in tables if str(t) in er_nodes},
                {uri_to_code(m): er_nodes[str(m)] for m in measures if str(m) in er_nodes},
                {uri_to_code(d): er_nodes[str(d)] for d in dims if str(d) in er_nodes})

    def get_candidate_nodes_bm25_ranked(self, query: str, k=25, table_cutoff: int = 5, obs_cutoff: int = 10) -> \
            (OrderedDict[str, Tuple[np.array, OrderedDict[str, np.array]]], Graph):
        try:
            g, tables, measures, dims = self.get_candidate_nodes_bm25(query, k)
        except Exception as e:
            logger.error(e)
            return OrderedDict(), None

        if g is None:
            logger.warning(f"Could not get subgraph for query {query}.")

        if len(tables) == 0 and g is None:
            logger.error(f"There were no tables nor subgraph for query {query}")
            return OrderedDict(), None

        explode_tables = list(tables.keys())[:table_cutoff]

        ranked_nodes = OrderedDict()
        for table in explode_tables or list({str(id_) for id_ in g.subjects(QB.measure, None)})[:table_cutoff]:
            ranked_nodes[table] = (None, OrderedDict())  # No embeddings are needed for this method. Default to None

            if g is not None:
                # Add measures and dims corresponding with table to the dict
                table_msrs = set(str(id_) for id_ in g.objects(TABLE.rdf_ns.term(table), QB.measure))
                table_dims = set(str(id_) for id_ in g.objects(TABLE.rdf_ns.term(table), QB.dimension))
            else:
                # Just add all measures and dimensions to every table if no graph could be generated
                table_msrs = set(measures)
                table_dims = set(dims)

            # Add measure candidates in beste scoring order to table
            sorted_msrs = sorted((set(measures) & table_msrs), key={k: d for k, (_, d) in measures.items()}.get)
            for msr in sorted_msrs[:obs_cutoff] or list(table_msrs)[:obs_cutoff]:
                ranked_nodes[table][1][msr] = None

            # Add dimension candidates in beste scoring order to table
            sorted_dims = sorted((set(dims) & table_dims), key={k: d for k, (_, d) in dims.items()}.get)
            for dim in sorted_dims[:obs_cutoff] or list(table_dims)[:obs_cutoff]:
                ranked_nodes[table][1][dim] = None

        return ranked_nodes, g

    def get_candidate_nodes_embeddings(self, query: str, full_graph: bool = False,
                                       faiss_k: int = 75, table_cutoff: int = 5, obs_cutoff: int = 10,
                                       verbose: bool = False) -> \
            (OrderedDict[str, Tuple[np.array, OrderedDict[str, np.array]]], Graph):
        """
            Get nodes from a Faiss-index, containing vector embeddings, based on
            the distance from the query. The returned dictionary contains the tables
            in order of L2 distance with the n closest measures and dimension based
            on a cutoff. Time- and GeoDimension are not added to the prompt to save
            space, as these can get large very quickly.

            :param query: the query to find nearest node matches on in the graph
            :param full_graph: fetch full graph, including dimension hierarchy relations (slower)
            :param faiss_k: number of closest Faiss-nodes to consider
            :param table_cutoff: maximum number of tables to explode to prevent overflowing of prompt
            :param obs_cutoff: maximum number of measures/dims (so x2) per table in prompt to prevent overflowing
            :param verbose: print the graph explosion SPARQL-query
            :returns: Tuple containing: (OrderedDict containing per table the best scoring measures and dimensions,
                                         exploded subgraph)
        """
        expanded_query = ' '.join(expand_query([w for w in word_tokenize(query) if w not in string.punctuation]))
        query_embedding = self.encoder.encode([expanded_query])

        # TODO: implement distance threshold
        dists, idx, embeddings = self.faiss_index.search_and_reconstruct(query_embedding, faiss_k)
        candidates: OrderedDict[str, (np.array, float)] = OrderedDict()
        for dist, idx, vec in zip(dists[0][::-1], idx[0][::-1], embeddings[0][::-1]):
            candidates[self.uri_map[str(idx)]] = (vec, dist)

        # Subdivide nodes in their respective categories
        tables = OrderedDict((k, v) for k, v in candidates.items() if k in TABLE.rdf_ns)
        measures = OrderedDict((k, v) for k, v in candidates.items() if k in MSR.rdf_ns)
        dims = OrderedDict((k, v) for k, v in candidates.items() if k in DIM.rdf_ns)

        # Explode tables to know the structure for the input prompt
        explode_tables = list(tables.keys())[:table_cutoff]
        explode_nodes = (explode_tables if len(explode_tables) > 0
                         else list(measures.keys()) + list(dims.keys()))
        try:
            expl_fun = explode_subgraph_msr_dims_only if not full_graph else explode_subgraph
            g = expl_fun(explode_nodes, table_cutoff, verbose)
        except Exception as e:
            logger.error(e)

        if g is None:
            logger.warning(f"Could not get subgraph for query {query}.")

        if len(tables) == 0 and g is None:
            return OrderedDict(), None

        # Get vectors for all candidates from the Faiss index. Takes the Faiss-retrieved
        # scored nodes if present, otherwise defaults to nodes from exploded subgraph.
        ranked_nodes: OrderedDict[str, Tuple[np.array, OrderedDict[str, np.array]]] = OrderedDict()
        for table in explode_tables or list({str(id_) for id_ in g.subjects(QB.measure, None)})[:table_cutoff]:
            if not (idx := self.inv_uri_map.get(table)):
                logger.warning(f"Table {table} not found in Faiss-index. Skipping...")
                continue

            ranked_nodes[table] = (
                tables.get(table, [self.faiss_index.reconstruct(int(idx))])[0],
                OrderedDict()
            )

            if g is not None:
                # Add measures and dims corresponding with table to the dict
                table_msrs = set(str(id_) for id_ in g.objects(URIRef(table), QB.measure))
                table_dims = set(str(id_) for id_ in g.objects(URIRef(table), QB.dimension))
            else:
                # Just add all measures and dimensions to every table if no graph could be generated
                table_msrs = set(measures)
                table_dims = set(dims)

            # Add measure candidates in beste scoring order to table
            sorted_msrs = sorted((set(measures) & table_msrs), key={k: d for k, (_, d) in measures.items()}.get)
            for msr in sorted_msrs[:obs_cutoff] or list(table_msrs)[:obs_cutoff]:
                if not (idx := self.inv_uri_map.get(msr)):
                    ranked_nodes[table][1][msr] = None
                    continue
                ranked_nodes[table][1][msr] = measures.get(msr, [self.faiss_index.reconstruct(int(idx))])[0]

            # Add dimension candidates in beste scoring order to table
            sorted_dims = sorted((set(dims) & table_dims), key={k: d for k, (_, d) in dims.items()}.get)
            for dim in sorted_dims[:obs_cutoff] or list(table_dims)[:obs_cutoff]:
                if not (idx := self.inv_uri_map.get(dim)):
                    ranked_nodes[table][1][dim] = None
                    continue
                ranked_nodes[table][1][dim] = dims.get(dim, [self.faiss_index.reconstruct(int(idx))])[0]

        return ranked_nodes, g
