import string
from collections import Counter
from nltk import word_tokenize
from rdflib import SKOS

import pipeline.logical_forms as logical_forms
from pipeline.candidate_ranking import CandidateRanker
from pipeline.entity_recognition import EntityRecognizer
from pipeline.fill_geo_constraints import match_region
from pipeline.fill_time_constraints import extract_tc
from pipeline.logical_forms import AGGREGATION, WHERE, MSR, DIM, OR, EOS, uri_to_code
from pipeline.odata_executor import ODataExecutor
from pipeline.query_expander_controller import expand_query
from pipeline.s_expression_util import SExpression


def baseline_bm25(query, return_sexp=False, verbose=False):
    er = EntityRecognizer()
    try:
        subgraph, tables, measures, dims = er.get_candidate_nodes_bm25(query, full_graph=True, verbose=verbose)

        if not tables or not measures:
            return None
    except Exception as e:
        raise f"Could not explode subgraph: {e}"

    # TODO: checken of sortering werkt
    cr = CandidateRanker(subgraph, measures, dims)
    r_measures, r_dims = cr.rank_entities_bm25()

    sexp = SExpression(subgraph)

    if verbose:
        print("Generating S-expression...")

    time_constraints = []
    geo_constraints = []
    table_groups = set()
    while 1:
        tokens = sexp.get_admissible_tokens()
        match len(tokens):
            case 0: break
            case 1:
                sexp.add_token(tokens[-1])
                continue

        # Choose TABLE node
        if isinstance(sexp._current_node, AGGREGATION):
            candidates = {t: tables[t]['score'] for t in (set(tables) & set(tokens))}
            if len(candidates) == 0:
                raise Exception("Could not generate S-expression. No suitable table found for the given query.")

            sexp.add_token(max(candidates, key=candidates.get))

            # Get valid dim groups for the chosen table based on the possible groups and dims we have scores for
            admissible_groups = {
                g: uri_to_code(sexp._tc_gc_to_dim_id(getattr(logical_forms, g, None))) if g in ['TC', 'GC'] else g
                for g in sexp.get_valid_dim_groups()
            }

            # Get applicable GC and TC clauses if admissible
            time_constraints = extract_tc(query)  # TODO: restrict with possible TCs only
            if 'TC' in admissible_groups and len(time_constraints) > 0:
                table_groups.add('TC')

            geo_constraints = match_region(query)
            if 'GC' in admissible_groups and len(geo_constraints) > 0:
                table_groups.add('GC')

            for dim in r_dims.keys():
                if dim in admissible_groups:  # filter out dim groups straight out of Elastic
                    continue

                group = list(subgraph.objects(DIM.rdf_ns.term(dim), SKOS.broader))[-1]
                if (dim_id := uri_to_code(group)) in admissible_groups:
                    table_groups.add(dim_id)
            continue

        # Choose MSR code node
        if isinstance(sexp._current_node, MSR) and len(sexp._current_node.children) == 0:
            # TODO: QUDT:Total measures in graaf opnemen en hyperparameter voor 'voorkeur' voor total measures
            #  (truc ook toepassen op dimensions)
            candidates = Counter({msr: r_measures[msr] for msr in (set(r_measures) & set(tokens))}).most_common()
            sexp.add_token(candidates[0][0] if candidates else tokens[-1])
            continue

        # Add WHERE node if applicable dims came out of the candidate ranker
        if WHERE.__name__ in tokens:
            sexp.add_token(WHERE.__name__ if len(table_groups) > 0 else EOS)
            continue

        # Add DIM node if applicable dims came out of the candidate ranker
        if DIM.__name__ in tokens:
            sexp.add_token(DIM.__name__ if len(table_groups) > 0 else EOS)
            continue

        # Add DIM code nodes based on applicable ones from the candidate ranker
        if isinstance(sexp._current_node, DIM) or isinstance(sexp._current_node, OR):
            if isinstance(sexp._current_node, DIM) and len(sexp._current_node.children) == 0:
                sexp.add_token(table_groups.pop())
            else:
                candidates = (
                        (set(r_dims) | set(['<TC>'] * len(time_constraints)) | set(['<GC>'] * len(geo_constraints)))
                        & set(tokens)
                )
                if len(candidates) == 0 and OR.__name__ in tokens and len(tokens) < 5:
                    # TODO: make proper fix. If no candidates but we have to choose do smth smart.
                    sexp.add_token(OR.__name__)
                elif len(candidates) == 0 and len(tokens) > 0:
                    sexp.add_token([t for t in tokens if t not in [OR.__name__, EOS]][0])
                elif len(candidates) == 0:
                    sexp.add_token(EOS)
                elif ((len(candidates) > 1 and OR.__name__ in tokens) or
                      (len(time_constraints) > 1 and '<TC>' in candidates) or
                      (len(geo_constraints) > 1 and '<GC>' in candidates)):
                    sexp.add_token(OR.__name__)
                elif candidates == {'<TC>'}:
                    sexp.add_token('<TC>')
                    time_constraints.pop()  # remove an arbitrary TC in order to prevent adding unlimited times <TC>
                elif candidates == {'<GC>'}:
                    sexp.add_token('<GC>')
                    geo_constraints.pop()  # remove an arbitrary GC in order to prevent adding unlimited times <GC>
                else:
                    code = Counter({dim: r_dims[dim] for dim in candidates}).most_common()[0][0]
                    sexp.add_token(code)
                    del r_dims[code]
                continue

    if return_sexp:
        return sexp

    if verbose:
        print(sexp)
        sexp.print_tree()

    odata = ODataExecutor(query, sexp, measures, dims)
    answer = odata.query_odata()
    answer.query = query
    answer.expanded_terms = expand_query([w for w in word_tokenize(query) if w not in string.punctuation])[:-1]

    return answer.sexp, answer


if __name__ == "__main__":
    # query = 'Hoeveel gaf de overheid uit aan zorg in 2020?'
    query = 'Huiselijk geweld; aard geweld, aanvullende informatie, regio'
    # query = 'Hoeveel mensen gingen er in 2021 met de trein op vakantie naar het buitenland?'
    # query = 'Internetgebruik van bedrijven'

    _, answer = baseline_bm25(query, verbose=True)
    print(answer)
