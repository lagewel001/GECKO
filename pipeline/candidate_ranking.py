from rdflib import Graph, RDF
import pandas as pd

from pipeline.logical_forms import MSR, DIM
from pipeline.sparql_controller import SCOT


class CandidateRanker:
    def __init__(self, graph: Graph, measures, dims):
        self.graph = graph
        self.measures = measures
        self.dims = dims

    def rank_entities_bm25(self, total_boost_factor=2, epsilon=0.1):
        """
            :param total_boost_factor: factoor to boost measures/dimensions of type scot:total with
            :param epsilon: minimum score per document
        """
        totals = set(self.graph.subjects(RDF.type, SCOT.Total))
        measures = {k: v['score'] * (total_boost_factor if MSR.rdf_ns.term(k) in totals else 1)
                    for k, v in self.measures.items() if v['score'] > epsilon}
        dims = {k: v['score'] * (total_boost_factor if DIM.rdf_ns.term(k) in totals else 1)
                for k, v in self.dims.items() if v['score'] > epsilon}

        # TODO: speed up w/ numpy
        ranked_msrs = {}
        msr_word_groups = pd.DataFrame([(k, word) for k, v in self.measures.items()
                                        for word in v['matched_words'] or []])
        if not msr_word_groups.empty:
            for matched_word, group in msr_word_groups.groupby(1):
                group['score'] = group[0].map(measures)
                top_msr = group.sort_values('score', ascending=False).iloc[0]
                ranked_msrs[top_msr[0]] = top_msr['score']

        ranked_dims = {}
        dim_word_groups = pd.DataFrame([(k, word) for k, v in self.dims.items() for word in v['matched_words'] or []])
        if not dim_word_groups.empty:
            for matched_word, group in dim_word_groups.groupby(1):
                group['score'] = group[0].map(dims)
                top_dim = group.sort_values('score', ascending=False).iloc[0]
                ranked_dims[top_dim[0]] = top_dim['score']

        # TODO: remove if not necessary. Sorts the dictionaries in case grouping on matched word proves useless
        # ranked_msrs = dict(sorted(measures.items(), reverse=True, key=lambda item: item[1]))
        # ranked_dims = dict(sorted(dims.items(), reverse=True, key=lambda item: item[1]))

        # TODO: experiment with removing all dims below a relative threshold score based on the top scoring msrs/dims
        return ranked_msrs, ranked_dims
