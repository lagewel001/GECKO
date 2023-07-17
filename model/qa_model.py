from transformers import AutoTokenizer

import paths_config
from model.encoderdecoder_trainer import SExpressionEncoderDecoderTrainer
from pipeline.entity_recognition import EntityRecognizer
from pipeline.logical_forms import uri_to_code, MSR, DIM
from pipeline.odata_executor import ODataExecutor


er = EntityRecognizer()
tokenizer = AutoTokenizer.from_pretrained(paths_config.TOKENIZER_EXT_PATH)
model = SExpressionEncoderDecoderTrainer(tokenizer=tokenizer,
                                         pretrained_or_checkpoint=f"{paths_config.TRAINED_MODEL_PATH}/model",
                                         beam_size=2)


def qa_model(query: str, return_sexp: bool = False, verbose: bool = False):
    sexp, ranked_nodes = model.conditional_generation(query, verbose)

    if return_sexp:
        return sexp

    if verbose:
        print(sexp)
        sexp.print_tree()

    try:
        ranked_msrs = [uri_to_code(o) for o in ranked_nodes.keys() if o in MSR.rdf_ns]
        ranked_dims = [uri_to_code(o) for o in ranked_nodes.keys() if o in DIM.rdf_ns]
        measures = dict(zip(ranked_msrs, range(len(ranked_msrs))))
        dims = dict(zip(ranked_msrs, range(len(ranked_dims))))

        odata = ODataExecutor(query, sexp, measures, dims)
        answer = odata.query_odata()
        answer.query = query

        return answer.sexp, answer
    except:
        return sexp, None


if __name__ == "__main__":
    # query = 'Uitgaven zorg overheid 2020'
    # query = 'Vervoer pijpleidingen in Nederland?'
    # query = 'Hoeveel mensen gingen er met de trein op vakantie naar het buitenland?'
    # query = 'Internetgebruik van bedrijven'
    query = 'Gemiddelde energieprijzen van consumenten in 2018'

    _, answer = qa_model(query, verbose=False)
    print(answer)
