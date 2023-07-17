import pandas as pd
import re
import time
from rdflib import Graph, ConjunctiveGraph, Namespace, Literal, RDF, DCTERMS as DCT, SKOS, URIRef, QB, SDO, DC, DCAT
from rdflib.term import Identifier, _is_valid_uri
from SPARQLWrapper import JSON
from tqdm import tqdm

import paths_config
from global_functions import secure_request
from pipeline.logical_forms import TABLE, MSR, DIM
from pipeline.sparql_controller import SCOT, sparql
from unit_convertor import UNIT_DICT, UNIT, MULTIPLIER, UNIT_OF_SYSTEM

url = "https://odata4.cbs.nl/CBS/Datasets"
table_list = secure_request(url, max_retries=3, timeout=3)

SDO._NS = Namespace("http://schema.org/")  # Op verzoek van Henk http:// i.p.v. https://

conj = ConjunctiveGraph()
TABLE_CTX = Graph(conj.store, URIRef("http://datasets-general_lucas"))
DIM_CTX = Graph(conj.store, URIRef("http://dimensies_lucas"))
QB_CTX = Graph(conj.store, URIRef("http://datasets-cube_lucas"))
MSR_CTX = Graph(conj.store, URIRef("http://onderwerpen_lucas"))
UNIT_CTX = Graph(conj.store, URIRef("http://units_lucas"))

triples = []
dimension_cache = []

# TODO: use URI's for geo dimensions
# TODO: use URI's for frequencies
# TODO: use property paths for traversing graph (https://en.wikibooks.org/wiki/SPARQL/Property_paths)

class Triple:
    def __init__(self, sub, pred, obj, ctx: Graph):
        self.sub = sub
        self.pred = pred
        self.obj = obj if isinstance(obj, Identifier) else Literal(obj)

        triples.append(self.__dict__)
        ctx.add((self.sub, self.pred, self.obj))


def uid(id_: str):
    id_ = id_.replace(' ', '')
    invalid_uri = not id_ or not _is_valid_uri(id_)
    if invalid_uri:
        return False
    return id_

msr_group_cache = []
measure_cache = []

def get_measures(table_id, table_node):
    """
        Add measure units and quantities
    """
    t0 = time.time()

    url = f"https://odata4.cbs.nl/CBS/{table_id}"
    table = [m['name'] for m in secure_request(url, max_retries=3, timeout=3)['value']]

    measures = []
    if 'MeasureCodes' in table:
        measure_codes = secure_request(f"{url}/MeasureCodes", max_retries=3, timeout=3)['value']
        for code in measure_codes:
            measures.append(code)

    if 'MeasureGroups' in table and 'MeasureCodes' in table:
        measure_groups = secure_request(f"{url}/MeasureGroups", max_retries=3, timeout=3)['value']
        for group in measure_groups:
            group_id = uid(group['Id'])

            # Add MeasureGroup triples
            if not group_id:
                print(f"MeasureGroup {group['Id']} for table_id {table_id} is not a valid id!")
                continue

            group_node = DIM.rdf_ns.term(group_id)
            Triple(group_node, DCT.isPartOf, table_node, MSR_CTX)  # Connect to table_id

            if group_node in msr_group_cache:
                continue

            Triple(group_node, RDF.type, QB.MeasureProperty, MSR_CTX)
            Triple(group_node, RDF.type, SKOS.Concept, MSR_CTX)
            Triple(group_node, SKOS.note, "MeasureGroup", MSR_CTX)
            Triple(group_node, DC.identifier, group['Id'], MSR_CTX)
            Triple(group_node, SKOS.inScheme, URIRef('https://vocabs.cbs.nl/def/onderwerpen/scheme'), MSR_CTX)
            Triple(group_node, SKOS.topConceptOf, URIRef('https://vocabs.cbs.nl/def/onderwerpen/scheme'), MSR_CTX)

            label = (SKOS.prefLabel if SKOS.prefLabel not in [t['pred'] for t in triples if t['sub'] == group_node]
                     else SKOS.altLabel)
            Triple(group_node, label, Literal(group['Title'], lang='nl'), MSR_CTX)

            if group.get('Description'):
                Triple(group_node, SKOS.definition, Literal(group['Description'].strip(), lang='nl'), MSR_CTX)

            if (p := group['ParentId']) is not None:
                if not (p_id := uid(p)):
                    print(f"Can't add {p} as parent to measure group {group_id}!")
                Triple(group_node, SKOS.narrower, DIM.rdf_ns.term(p_id), MSR_CTX)
                Triple(DIM.rdf_ns.term(p_id), SKOS.broader, group_node, MSR_CTX)

            msr_group_cache.append(group_node)

    for msr in measures:
        if not (msr_id := uid(msr.get('Identifier'))):
            print(f"Measure {msr.get('Identifier')} for table_id {table_id} is not a valid id!")
            continue

        # Add Measure triples
        msr_node = MSR.rdf_ns.term(msr_id)
        Triple(table_node, QB.measure, msr_node, MSR_CTX)  # Connect to table_id

        if msr_node in measure_cache:
            continue

        Triple(msr_node, RDF.type, QB.MeasureProperty, MSR_CTX)
        Triple(msr_node, RDF.type, SKOS.Concept, MSR_CTX)
        Triple(msr_node, SKOS.note, "Measure", MSR_CTX)
        Triple(msr_node, DC.identifier, msr['Identifier'], MSR_CTX)
        Triple(msr_node, SKOS.inScheme, URIRef('https://vocabs.cbs.nl/def/onderwerpen/scheme'), MSR_CTX)

        label = (SKOS.prefLabel if SKOS.prefLabel not in [t['pred'] for t in triples if t['sub'] == msr_node]
                 else SKOS.altLabel)
        Triple(msr_node, label, Literal(msr['Title'], lang='nl'), MSR_CTX)

        measure_cache.append(msr_node)

        if msr.get('Description'):
            Triple(msr_node, SKOS.definition, Literal(msr['Description'].strip(), lang='nl'), MSR_CTX)

        if (p := msr['MeasureGroupId']) is not None:
            if not (p_id := uid(p)):
                print(f"Can't add {p} as parent to measure {msr_id}!")
            Triple(msr_node, SKOS.narrower, DIM.rdf_ns.term(p_id), MSR_CTX)
            Triple(MSR.rdf_ns.term(p_id), SKOS.broader, msr_node, MSR_CTX)

        if re.match(r".*\b(totaal|waarde|totale)\b.*", msr['Title'], flags=re.IGNORECASE):
            Triple(msr_node, RDF.type, SCOT.Total, UNIT_CTX)
        if re.match(r".*\b(interval)\b.*", msr['Title'], flags=re.IGNORECASE):
            Triple(msr_node, RDF.type, SCOT.confidenceInterval, UNIT_CTX)

        if msr.get('Unit') and UNIT_DICT.get(msr['Unit'], False):
            unit = UNIT_DICT[msr['Unit']]
            Triple(msr_node, UNIT_OF_SYSTEM, msr['Unit'], UNIT_CTX)
            if unit['unit'] is not None:
                Triple(msr_node, UNIT, unit['unit'], UNIT_CTX)
            if unit['multiplier'] is not None:
                Triple(msr_node, MULTIPLIER, Literal(unit['multiplier']), UNIT_CTX)

            if re.match(r".*\b(totaal|waarde|totale)\b.*", msr['Title'], flags=re.IGNORECASE):
                Triple(msr_node, RDF.type, SCOT.Total, UNIT_CTX)
            if re.match(r".*\b(interval)\b.*", msr['Title'], flags=re.IGNORECASE):
                Triple(msr_node, RDF.type, SCOT.confidenceInterval, UNIT_CTX)

    print(f"\nTime to get measures for table {table_id}: {time.time() - t0}")
    return measures


def get_dimensions(table_id, table_node):
    """
        Hiërarchie:
            Tabel -> Dimensies -> Groups -> Codes

        Dimensions types: "Dimension", "GeoDimension", "GeoDetailDimension", "TimeDimension"
    """
    # Generate dimension dictionary
    url = f"https://odata4.cbs.nl/CBS/{table_id}/Dimensions"
    dims = secure_request(url, max_retries=3, timeout=3)['value']
    table_dimensions = []
    for dim in dims:
        dim['SubDimensions'] = []
        if dim['ContainsGroups'] and dim['ContainsCodes']:
            dim['Groups'] = []
            dimension_groups = secure_request(dim['GroupsUrl'], max_retries=3, timeout=3)['value']
            dimension_codes = secure_request(dim['CodesUrl'], max_retries=3, timeout=3)['value']
            for group in dimension_groups:
                group['SubDimensions'] = []
                group_id = group['Id']
                for code in dimension_codes:
                    if not code['DimensionGroupId']:
                        dim['SubDimensions'].append(code)
                    if group_id == code['DimensionGroupId']:
                        group['SubDimensions'].append(code)
                dim['Groups'].append(group)
        elif dim['ContainsCodes']:
            dimension_codes = secure_request(dim['CodesUrl'], max_retries=3, timeout=3)['value']
            dim['SubDimensions'] = []
            for code in dimension_codes:
                dim['SubDimensions'].append(code)
        table_dimensions.append(dim)

    def _add_dim_triples(dim, kind=None):
        id_ = uid(dim['Identifier'])
        if not id_:
            print(f"Dimension {dim['Identifier']} for table_id {table_id} is not a valid id!")

        node = DIM.rdf_ns.term(id_)

        # Connect to table_id
        Triple(table_node, QB.dimension, node, QB_CTX)

        if node in dimension_cache:
            return node

        Triple(node, DCT.identifier, dim['Identifier'], DIM_CTX)
        Triple(node, RDF.type, QB.DimensionProperty, DIM_CTX)
        Triple(node, SDO.isBasedOn, "Dimensies", DIM_CTX)
        Triple(node, SKOS.prefLabel, Literal(dim['Title'], lang='nl'), DIM_CTX)
        if dim['Description']:
            Triple(node, DCT.description, Literal(dim['Description'], lang='nl'), DIM_CTX)

        if kind == 'TimeDimension':
            # TODO: do something official like dct:spatial or dct:temporal instead of using Literal
            Triple(node, RDF.type, 'TimeDimension', DIM_CTX)
        if kind in ['GeoDimension', 'GeoDetailDimension']:
            Triple(node, RDF.type, 'GeoDimension', DIM_CTX)

        if re.match(r".*\b(totaal|waarde|totale)\b.*", dim['Title'], flags=re.IGNORECASE):
            Triple(node, RDF.type, SCOT.Total, UNIT_CTX)
        if re.match(r".*\b(interval)\b.*", dim['Title'], flags=re.IGNORECASE):
            Triple(node, RDF.type, SCOT.confidenceInterval, UNIT_CTX)

        dimension_cache.append(node)
        return node

    for dim in table_dimensions:
        kind = dim.get('Kind')
        dim_node = _add_dim_triples(dim, kind)

        codes = dim['SubDimensions']  # Codes hanging directly under dimension (without group)
        for grp in dim.get('Groups', []):
            for subdim in grp['SubDimensions']:
                codes.append(subdim)

        for code in codes:
            code_node = _add_dim_triples(code, kind)
            Triple(dim_node, SKOS.narrower, code_node, DIM_CTX)
            Triple(code_node, SKOS.broader, dim_node, DIM_CTX)

    return dims


OVERWRITE = False
REPOSITORY = paths_config.GRAPH_DB_REPO

tables = secure_request('https://odata4.cbs.nl/CBS/datasets', json=True, max_retries=3, timeout=3)['value']

questions = pd.read_csv(paths_config.DATASET_PATH, delimiter=';').sexp
question_tables = questions.str.extract(r'(?<=\(VALUE \()(.*?)(?=\s)', expand=False).unique()

kerncijfer_tables = [t['Identifier'] for t in tables
                     if ('kerncijfers' in t['Title'].lower() and t['Status'] == 'Regulier')
                     or t['Identifier'] in question_tables]
non_active_kerncijfers = [t['Identifier'] for t in tables if 'kerncijfers' in t['Title'].lower() and t['Status'] != 'Regulier']

if table_list is False:
    stop = 1
else:
    for table_id in tqdm(set(kerncijfer_tables), desc="*Prrt prrrt* Generating RDF triples for tables...",
                         bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
        table_node = TABLE.rdf_ns.term(table_id)

        # Check if table_id exists in graph
        sparql.setReturnFormat(JSON)
        query = (f"""
            PREFIX dct: <http://purl.org/dc/terms/>
            SELECT ?id WHERE {{ <{table_node}> dct:identifier ?id . }}
        """)
        sparql.setQuery(query)
        sparql.method = 'GET'

        try:
            if OVERWRITE or len(sparql.queryAndConvert()['results']['bindings']) == 0:
                # print(f"TABLE {table_node} NOT IN GRAPH!")
                props = secure_request(f"https://odata4.cbs.nl/CBS/{table_id}/Properties", max_retries=3, timeout=3)
                Triple(table_node, RDF.type, DCAT.Dataset, TABLE_CTX)
                # Triple(table_node, DCT.accrualPeriodicity, URIRef(f"https://vocabs.cbs.nl/def/frequenties/{en_props['Frequency'].lower()}"), TABLE_CTX)  # TODO
                # Triple(table_node, DCT.accrualPolicy, URIRef(f"https://vocabs.cbs.nl/def/frequenties/{en_props['Frequency'].lower()}"), TABLE_CTX)
                Triple(table_node, DCT.identifier, props['Identifier'], TABLE_CTX)
                Triple(table_node, DCT.modified, props['Modified'], TABLE_CTX)
                Triple(table_node, DCT.source, props['Catalog'], TABLE_CTX)
                # Triple(table_node, DCT.spatial, None, TABLE_CTX)  # TODO
                Triple(table_node, DCT.coverage, props['TemporalCoverage'], TABLE_CTX)
                # Triple(table_node, DCT.temporal, None, TABLE_CTX)
                Triple(table_node, DCT.title, Literal(props['Title'], lang='nl'), TABLE_CTX)
                Triple(table_node, URIRef('http://purl.org/linked-data/sdmx/2009/attribute#freqDiss'), None, TABLE_CTX)
                Triple(table_node, DCAT.catalog, URIRef('https://opendata.cbs.nl/#/CBS/nl/dataset'), TABLE_CTX)

                table_period = re.findall(r'\b[1,2]\d{3}\b', props['TemporalCoverage'])
                if len(table_period) > 1:
                    table_period = table_period[::len(table_period) - 1]
                Triple(table_node, DCAT.startDate, table_period[0], TABLE_CTX)
                Triple(table_node, DCAT.endDate, table_period[-1], TABLE_CTX)
                # Triple(table_node, DCAT.keyword, None, TABLE_CTX)  # TODO

                if props.get('Description'):
                    Triple(table_node, DCT.description, Literal(props['Description'], lang='nl'), TABLE_CTX)

                measures = get_measures(table_id, table_node)
                dims = get_dimensions(table_id, table_node)
        except Exception as e:
            print(f"Failed to fetch table_id nodes: {e}")

triples = pd.DataFrame(triples, columns=['sub', 'pred', 'obj'])

graphs = [
    ('datasets.trig', TABLE_CTX.serialize(format='trig')),
    ('datasets-cube.trig', QB_CTX.serialize(format='trig')),
    ('dimensies.trig', DIM_CTX.serialize(format='trig')),
    ('onderwerpen.trig', MSR_CTX.serialize(format='trig')),
    ('units.trig', UNIT_CTX.serialize(format='trig')),
]

output_dir = './graph/'
for file_name, ctx in graphs:
    with open(f"{output_dir}{file_name}", 'wb') as file:
       file.write(ctx.encode())
