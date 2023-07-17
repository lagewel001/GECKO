# GECKO

## Preliminaries:
- Elasticsearch is running locally at port 9200 (or online with correct connection string added as `host` in [elastic.py](./elastic.py))
- GraphDB and a repository are running online or locally and connection information is set in [sparql_controller.py](./pipeline/sparql_controller.py)
- Python venv (>=3.10) is up and running and [requirements](./requirements.txt) are installed
  - [NLTK 'punkt' and 'stopwords' are downloaded](https://www.nltk.org/data.html)
- Pretrained model downloads when running model-based GECKO:
  1. [SNERT_PLM.zip](https://vunl-my.sharepoint.com/:f:/g/personal/l_lageweg_student_vu_nl/EohyW52LOLtIsHG2QzBau34B7rEVuGQ5gliio5CKKtbqXQ?e=f5Z0Z4); extract folder to `./model/SNERT_PLM`
  2. [SNERT_tokenizer.zip](https://vunl-my.sharepoint.com/:f:/g/personal/l_lageweg_student_vu_nl/EohyW52LOLtIsHG2QzBau34B7rEVuGQ5gliio5CKKtbqXQ?e=f5Z0Z4); extract folder to `./model/SNERT_tokenizer`
  3. [SNERTe_model.zip](https://vunl-my.sharepoint.com/:f:/g/personal/l_lageweg_student_vu_nl/EohyW52LOLtIsHG2QzBau34B7rEVuGQ5gliio5CKKtbqXQ?e=f5Z0Z4); extract folder to `./model/SNERTe_model`
- Correct paths are set in [paths_config.py](./paths_config.py) for the configuration that you want to run.

Every script should be run with the project root as working directory. File and folder paths can be configured through 
[paths_config.py](./paths_config.py).


## Running for the first time
When running GECKO for the first time, a few things need to be set up.

### Case: GraphDB has not been filled yet.
- Run [odata_rdf_generator.py](./graph/odata4_rdf_generator.py). Upload/import the resulting `.trig` files in the `/graph` folder to your GraphDB repository. 
- Set the correct repository name in [paths_config.py](./paths_config.py).

### Case: Elastic has not been indexed yet OR for the model configuration I want to run there is no Faiss-index yet in the `/data` directory.
Set the correct paths in [paths_config.py](./paths_config.py) for the pre-trained model you want to run (if applicable). Run [graph_es_indexer.py](./graph/graph-es-indexer.py).


## Training, inference and evaluation
- For training, [encoderdecoder_trainer.py](./model/encoderdecoder_trainer.py) can be run.
- For inference run
  - baseline: [qa_baseline.py](./model/qa_baseline.py)
  - model: [qa_model.py](./model/qa_model.py) (pretrained model and Faiss-index required!)
- For evaluation run [evaluate_model.py](./model/evaluate_model.py) and configure in the file if you want to evaluate baseline or model-based GECKO.
