import copy
import datetime
import pandas as pd
from functools import cmp_to_key
from rdflib import Literal, RDF, SKOS, DCTERMS as DCT
from typing import Set, Dict

from pipeline.fill_geo_constraints import match_region
from pipeline.fill_time_constraints import extract_tc
from pipeline.logical_forms import *
from pipeline.s_expression_util import SExpression
from pipeline.sparql_controller import get_table_geo_dims, SCOT, QUDT, get_table_time_dims
from global_functions import secure_request


class QuestionAnswer:
    query: str
    sexp: SExpression
    observations: pd.DataFrame
    expanded_terms: Union[None, List[str]]
    table_match: str
    msr_matches: Dict[str, str]
    dim_matches: Dict[str, str]
    assumptions: Dict[str, str]

    def __init__(
            self,
            query: str = None,
            sexp: SExpression = None,
            expanded_terms: Union[None, List[str]] = None,
            observations: Optional[pd.DataFrame] = None,
            table_match: str = None,
            msr_matches: Dict[str, str] = None,
            dim_matches: Dict[str, str] = None,
            assumptions: Dict[str, str] = None
    ):
        self.query = query
        self.sexp = sexp
        self.expanded_terms = expanded_terms
        self.observations = observations
        self.table_match = table_match
        self.msr_matches = msr_matches
        self.dim_matches = dim_matches
        self.assumptions = assumptions

    def __str__(self):
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3):
            answer = ''
            answer += f"Query: {self.query}\n"
            answer += f"Generated S-expression: {str(self.sexp)}\n"

            if self.expanded_terms is not None:
                answer += f"Also searched on: {self.expanded_terms}\n"

            answer += '\n'
            answer += self.table_match + '\n'
            if self.observations is None:
                answer += "No value in OData4 for selected measure(s) and dimension(s). \n"
            else:
                answer += self.observations.to_markdown() + '\n'
            answer += '\n'

            for msr, match in self.msr_matches.items():
                answer += f"Matched {msr} on {match}\n"

            for dim, match in self.dim_matches.items():
                answer += f"Matched {dim} on {match}\n"

            for dim, match in self.assumptions.items():
                answer += f"Assumed {dim} to be searched on {match}\n"

            return answer


class QueryBuilder:
    """
        Helper class for generating a OData4 request/query to obtain
        observations of a table using given measure and dimension filters.
    """
    query = f"https://odata4.cbs.nl/CBS/"
    filters = False

    def __init__(self, table_id: TABLE):
        self.query += f"{table_id}/Observations/"

    def add_msr_filter(self, measures: Set[Code]):
        self.query += '?$filter=' if not self.filters else ' and '
        self.query += f'''Measure in ('{"', '".join([str(m) for m in measures])}')'''
        self.filters = True

    def add_dim_filter(self, group: DimId, codes: Set[Code]):
        self.query += '?$filter=' if not self.filters else ' and '
        # There's no distinction between codes of given dim group and all codes, but that shouldn't matter
        self.query += f'''{group} in ('{"', '".join([str(dim) for dim in codes])}')'''
        self.filters = True

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.query


class ODataExecutor:
    sexp: SExpression

    def __init__(self, query: str, sexp: SExpression, measures: dict, dims: dict):
        self.query = query
        self.sexp = sexp
        self.measures = measures
        self.dims = dims

    @staticmethod
    def _compare_time_dims(a, b):
        a = str(a).split(DIM.rdf_ns)[-1]
        b = str(b).split(DIM.rdf_ns)[-1]

        pattern = r"(?mx)(?P<year>\d{4})(?P<unit>[A-Z]{2}|X0)(?P<val>\d{2})"
        a_terms = re.match(pattern, a).groupdict()
        b_terms = re.match(pattern, b).groupdict()

        if a_terms['year'] > b_terms['year']:
            return 1
        if a_terms['year'] < b_terms['year']:
            return -1

        unit_order = ['X0', 'JJ', 'HJ', 'VJ', 'KW', 'MM']
        if unit_order.index(a_terms['unit']) < unit_order.index(b_terms['unit']):
            return 1
        if unit_order.index(a_terms['unit']) > unit_order.index(b_terms['unit']):
            return -1

        if a_terms['val'] > b_terms['val']:
            return 1
        if a_terms['val'] < b_terms['val']:
            return -1

        return 0

    @staticmethod
    def _compare_geo_dims(a, b):
        a = str(a).split(DIM.rdf_ns)[-1]
        b = str(b).split(DIM.rdf_ns)[-1]

        pattern = r"(?mx)(?P<unit>[A-Z]{2})(?P<code>\d{2,4})"
        a_terms = re.match(pattern, a).groupdict()
        b_terms = re.match(pattern, b).groupdict()

        unit_order = ['NL', 'LD', 'CR', 'CP', 'VT', 'PV', 'GM', 'BU', 'WK']
        if unit_order.index(a_terms['unit']) < unit_order.index(b_terms['unit']):
            return 1
        if unit_order.index(a_terms['unit']) > unit_order.index(b_terms['unit']):
            return -1

        # Filtering on code is quite redundant for regions
        return 0

    def _get_default_tc(self) -> (URIRef, list):
        """Get URI of TC dim group and corresponding default highest aggregated dim"""
        # TODO: TC's not fetched for table 70895ned for question `Overledenen; geslacht en leeftijd, per week`
        uri = (set(self.sexp.graph.subjects(RDF.type, Literal('TimeDimension'))) -
               set(self.sexp.graph.objects(None, SKOS.broader))).pop()
        dims = get_table_time_dims(self.sexp._current_table)

        cur_year = datetime.date.today().year
        sorted_keys = sorted(dims.keys(), reverse=True, key=cmp_to_key(self._compare_time_dims))

        # Pick the largest year up until current year, or smallest if first year is bigger than current year
        if int(sorted_keys[-1][:4]) > cur_year:
            return uri, sorted_keys[-1:]
        else:
            sorted_dims = [code for code in sorted_keys if int(code[:4]) <= cur_year]
            return uri, sorted_dims[:1]

    def _get_default_gc(self) -> (URIRef, list):
        """Get URI of GC dim group and corresponding default highest aggregated dim"""
        uri = (set(self.sexp.graph.subjects(RDF.type, Literal('GeoDimension'))) -
               set(self.sexp.graph.objects(None, SKOS.broader))).pop()
        dims = get_table_geo_dims(self.sexp._current_table)
        highest_agg_dims = sorted(dims.keys(), reverse=True, key=cmp_to_key(self._compare_geo_dims))[:1]

        return uri, highest_agg_dims

    def _get_default_dims(self) -> (dict, set):
        """
            Pick default dimensions for the missing dimension from the graph with the following rules:
            Dimension type:
                - TimeDimension (TC) -> pick most recent year/date possible;
                - GeoDimension (GC)  -> pick highest aggregation (country, province);
                - Other:             -> pick Total/Totaal if available for the specific dimension group.
            If no default dimension can be made the dimension group is returned again as 'missing'.
        """
        missing_dims = self.sexp.get_valid_dim_groups()  # dim_groups still valid are not filled in by the S-expression
        default_dims = {}
        for dim_group in copy.deepcopy(missing_dims):
            match dim_group:
                case TC.__name__:
                    uri, highest_agg_dims = self._get_default_tc()
                case GC.__name__:
                    uri, highest_agg_dims = self._get_default_gc()
                case _:
                    uri = DIM.rdf_ns.term(dim_group)
                    dims = {DIM.rdf_ns.term(d) for d in self.sexp.get_valid_dim_tokens(uri)}
                    highest_agg_dims = dims & set(self.sexp.graph.subjects(RDF.type, SCOT.Total))  # TODO: do we need to pick [0]?

            if highest_agg_dims:
                default_dims[str(uri).split(DIM.rdf_ns)[-1]] = [str(d).split(DIM.rdf_ns)[-1] for d in highest_agg_dims]
                missing_dims.remove(dim_group)

        return default_dims, missing_dims

    def query_odata(self) -> QuestionAnswer:
        default_dims, missing_dims = self._get_default_dims()

        if len(missing_dims) > 0:
            raise ValueError(f"Could not send OData-request. Missing dimensions "
                             f"{missing_dims} for expression {self.sexp}.")

        table = self.sexp._current_table
        obs_map = self.sexp.obs_map[table]

        # Substitute <GC>/<TC> placeholders
        for dim in obs_map.dims:
            if dim == '<TC>':
                # Extract constraint from query, default to highest dimension from table
                # TODO: only one Time and Geo constraint can now be replaced
                if not (tcs := extract_tc(self.query)):  # TODO: restrict with possible TCs
                    uri, tcs = self._get_default_tc()
                    if uri:
                        default_dims[str(uri).split(DIM.rdf_ns)[-1]] = \
                            [str(d).split(DIM.rdf_ns)[-1] for d in tcs]
                if len(tcs) > 0:
                    dim.value = tcs[0]
                    self.sexp.expression = self.sexp.expression.replace('<TC>', dim.value)

            if dim == '<GC>':
                if not (gcs := match_region(self.query)):
                    uri, gcs = self._get_default_gc()
                    if uri:
                        default_dims[str(uri).split(DIM.rdf_ns)[-1]] = \
                            [str(d).split(DIM.rdf_ns)[-1] for d in gcs]
                if len(gcs) > 0:
                    dim.value = gcs[0]
                    self.sexp.expression = self.sexp.expression.replace('<GC>', dim.value)

        odata_query = QueryBuilder(table)
        odata_query.add_msr_filter(obs_map.measures)
        for dim_group in obs_map.dim_groups:
            odata_query.add_dim_filter(dim_group, obs_map.dims)
        for dim_group, codes in default_dims.items():
            odata_query.add_dim_filter(dim_group, codes)

        try:
            observations = secure_request(str(odata_query), json=True, max_retries=9, timeout=3)['value']

            result_df = None  # Return no answer if observations is empty
            code_labels = {str(obs).split(MSR.rdf_ns if obs in MSR.rdf_ns else DIM.rdf_ns)[-1]: str(label)
                           for obs, label in self.sexp.graph.subject_objects(SKOS.prefLabel)}
            if len(default_dims) > 0:
                code_labels |= get_table_geo_dims(self.sexp._current_table)
                code_labels |= get_table_time_dims(self.sexp._current_table)
            if len(observations) > 0:
                # TODO: certain measures don't have values for certain periods (e.g. 81628NED/ATC1125/2022
                #  => https://opendata.cbs.nl/statline#/CBS/nl/dataset/81628NED/table?ts=1679387089708)
                obs_df = pd.DataFrame(observations)
                obs_df = obs_df.set_index('Id', drop=True)
                # TODO: als ValueAttribute niet None is betekent het dat we de waarde niet terug mogen geven -> terugkoppelen aan gebruiker
                obs_df = obs_df.drop(['ValueAttribute', 'StringValue'], axis=1)

                units = {uri_to_code(m): str(u) for m, u in self.sexp.graph.subject_objects(QUDT.unitOfSystem)}
                obs_df['Unit'] = obs_df['Measure'].map(units)

                obs_df = obs_df.rename(columns=code_labels)
                for c in obs_df.columns:
                    obs_df[c] = obs_df[c].replace(code_labels)

                result_df = obs_df.pivot(index=['Measure', 'Unit'],
                                         columns=[c for c in obs_df if c not in ['Measure', 'Value', 'Unit']],
                                         values='Value')

            table_match = f"{table} - {str(list(self.sexp.graph.objects(table.uri, DCT.title))[0])}"
            msr_matches = dim_matches = {}  # TODO: fix
            # msr_matches = {code_labels[str(m)]: self.measures[str(m)]['matched_words'] for m in obs_map.measures}
            # dim_matches = {code_labels[str(d)]: self.dims[str(d)]['matched_words'] for d in obs_map.dims}
            assumptions = {code_labels[dim]: [code_labels[c] for c in codes] for dim, codes in default_dims.items()}

            answer = QuestionAnswer(sexp=self.sexp,
                                    observations=result_df, table_match=table_match,
                                    msr_matches=msr_matches, dim_matches=dim_matches,
                                    assumptions=assumptions)
        except Exception as e:
            raise ValueError(f"Failed to retrieve data from OData4 request: {str(e)}")

        return answer
