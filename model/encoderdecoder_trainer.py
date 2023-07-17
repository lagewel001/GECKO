import csv
import evaluate
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import re
import time
import torch
from accelerate import Accelerator
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup, PreTrainedTokenizerFast, RobertaTokenizerFast,
    AutoConfig, RobertaPreTrainedModel,
    StoppingCriteria, StoppingCriteriaList, AutoTokenizer
)
from typing import Optional, Dict

import paths_config
from model.s_expression_encoderdecoder import SExpressionEncoderDecoder
from pipeline.entity_recognition import EntityRecognizer
from pipeline import logical_forms
from pipeline.logical_forms import (
    uri_to_code, SOS, EOS, OR, AGGREGATION,
    TABLE, MSR, DIM, TerminalNode, WHERE
)
from pipeline.s_expression_util import SExpression

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class HFDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class SExpressionStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        """Stop generating if all bracket of generated S-expression match."""
        beam_criteria = []
        for beam in input_ids:
            expression = self.tokenizer.decode(beam, skip_special_tokens=True)
            beam_criteria.append(expression.count(SOS) - expression.count(EOS) == 0)

        return all(beam_criteria)


class SExpressionEncoderDecoderTrainer:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            encoder_path: Optional[str] = None,
            decoder_path: Optional[str] = None,
            pretrained_or_checkpoint: Optional[str] = None,
            faiss_solution: int = 1,
            batch_size: int = 4,
            warmup_steps: float = 1e2,
            learning_rate: float = 3e-5,
            gradient_accumulation_steps: int = 1,
            special_tokens_weights_factor: float = 0.1,
            label_smoothing: float = 0.1,
            eval_steps: int = 100,
            seed: int = 42,
            epochs: int = 500,
            beam_size: int = 2
    ):
        self.tokenizer = tokenizer
        self.entity_retriever = EntityRecognizer()

        accelerator = Accelerator()
        self.device = accelerator.device

        # Prepare weights for special tokens in loss function
        loss_weights = torch.ones((len(tokenizer),))
        loss_weights[np.unique(self.tokenizer.encode('( ()', add_special_tokens=False) +
                               self.tokenizer.all_special_ids)] = special_tokens_weights_factor

        self.checkpoint = None
        assert (encoder_path and decoder_path) or pretrained_or_checkpoint
        if pretrained_or_checkpoint:
            # Load in model from checkpoint if applicable
            try:
                self.model = SExpressionEncoderDecoder.from_pretrained(pretrained_or_checkpoint,
                                                                       special_tokens_weights=loss_weights,
                                                                       label_smoothing=label_smoothing)
                self.model = accelerator.prepare(self.model)

                self.checkpoint = pretrained_or_checkpoint
                logger.info(f"Loaded pre-trained model from checkpoint {pretrained_or_checkpoint}. This model "
                            f"should be used for inference or continuing training from this checkpoint.")
            except OSError as e:  # TODO: fetch correct exception type
                raise FileNotFoundError(f"Can't find pretrained SExpressionDecoder model "
                                        f"checkpoint locally in path {pretrained_or_checkpoint}: {e}")
        else:
            logger.info(f"Initialising model from pretrained encoder {encoder_path} and decoder {decoder_path}.")
            encoder_config = AutoConfig.from_pretrained(encoder_path)
            decoder_config = AutoConfig.from_pretrained(decoder_path, is_decoder=True, add_cross_attention=True)
            self.model = SExpressionEncoderDecoder.from_encoder_decoder_pretrained(
                encoder_path, decoder_path,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                faiss_solution=faiss_solution,
                special_tokens_weights=loss_weights,
                label_smoothing=label_smoothing
            )

        # Resize default token embeddings to accommodate for added tokens to vocabulary
        if (len(self.tokenizer) != self.model.config.encoder.vocab_size or
            len(self.tokenizer) != self.model.config.encoder.vocab_size):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))
            self.model.decoder.resize_token_embeddings(len(self.tokenizer))

            if self.model.config.faiss_solution == 3:
                self.model.encoder.embeddings.initialize_faiss_weights()
                if isinstance(self.model.decoder, RobertaPreTrainedModel):
                    self.model.decoder.roberta.embeddings.initialize_faiss_weights()
                else:
                    self.model.decoder.bert.embeddings.initialize_faiss_weights()

                if self.model.loss_fct.weight is None:
                    self.model.loss_fct = torch.nn.CrossEntropyLoss(weight=loss_weights,
                                                                    label_smoothing=label_smoothing)

        self.model.to(self.device)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())

        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.seed = seed

        self.beam_size = beam_size
        self.max_source_length = 512
        self.max_target_length = 64  # TODO: what is reasonable?

        self.model.config.decoder_start_token_id = tokenizer.cls_token_id
        self.model.config.eos_token_id = tokenizer.sep_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id

        # Evaluation metrics
        self.bleu = evaluate.load("sacrebleu")
        self.rouge = evaluate.load("rouge")

    def generate_prompt(self, query: str, ranked_nodes: OrderedDict) -> str:
        """
            Generated prompt based on query string and nodes from Faiss

            :param query: query string as start of input prompt
            :param ranked_nodes: ordered nodes following from Faiss based on the query
            :returns: prompt for a single query in the form of
                      <query>[SEP]<table 1>|MSR|<msr 1>;<msr 2>|DIM|<dim 1>;<dim 2>;<dim 3>[SEP]<table 2>|MSR| ...
        """
        prompt = str(query)

        for table, (table_embedding, obs) in ranked_nodes.items():
            prompt += f"{self.tokenizer.sep_token}{uri_to_code(table)}"

            msrs = [uri_to_code(o) for o in obs if o in MSR.rdf_ns]
            if len(msrs) > 0:
                prompt += f"|MSR|{';'.join(msr for msr in msrs)}"
            else:
                logger.error(f"Encountered a prompt without a measure for query {query}. "
                             f"This shouldn't be possible.")

            # TODO: also partition by dimensions groups
            dims = [uri_to_code(o) for o in obs if o in DIM.rdf_ns]
            if len(dims) > 0:
                prompt += f"|DIM|{';'.join(msr for msr in dims)}"

        return prompt

    def generate_input_sequences(self, data: pd.DataFrame, pid: int) -> (list[str], list[str]):
        """
            Generate input and output sequences based on query-sexp pairs

            :param data: DataFrame containing query-sexp pairs
            :param pid: process ID indicating a multiprocessing process
            :returns: tuple containing list of input sequences and list of output sequences respectively
        """
        cheat_score = 0  # number of prompts were the target table did not yield from the entity retriever
        input_sequences = []
        output_sequences = []
        for _, row in tqdm(data.iterrows(), total=data.shape[0],
                           desc=f"Encoding input prompts (pid #{pid})".zfill(3), position=pid + 1,
                           bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
            target = row.sexp
            query = row.query

            if not target:
                logger.warning(f"S-expression is null for question {row.qid}")
                continue

            if not query or not isinstance(query, str):
                logger.warning(f"Query is null for question {row.qid}")
                continue

            # Substitute GC and TC codes for placeholders. These will be
            # filled in based on rules when parsing the S-expression.
            target = re.sub(r'GC \w+\b', 'GC <GC>', target)
            target = re.sub(r'TC \w+\b', 'TC <TC>', target)

            # Get candidate nodes following from the subgraph and the Faiss-index
            # ranked_nodes, g = self.entity_retriever.get_candidate_nodes_bm25_ranked(query, k=75)
            ranked_nodes, g = self.entity_retriever.get_candidate_nodes_embeddings(query, faiss_k=75)
            if len(ranked_nodes) == 0:
                continue

            # Add nodes from the S-expression if they're missing from the ER-step (cheat for training)
            # TODO: evaluate if this is good for the model when we have a lot of training data
            try:
                sexp = SExpression.parse(target)
                uri = str(TABLE.rdf_ns.term(str(sexp._current_table)))

                obs = sexp.obs_map[sexp._current_table]
                obs = {o for o in obs.measures | obs.dims | obs.dim_groups if o.value not in ['<TC>', '<GC>']}

                if uri not in ranked_nodes and str(uri) in self.entity_retriever.inv_uri_map:
                    ranked_nodes[uri] = (
                        None,  # no need to do faiss.reconstruct here
                        OrderedDict((str(o.uri), None) for o in obs)
                    )

                    ranked_nodes.move_to_end(uri, last=False)
                    cheat_score += 1
                else:
                    missing_obs = {str(o.uri) for o in obs} - set(ranked_nodes[uri][1])
                    ranked_nodes[uri][1].update({o: None for o in missing_obs})
                    if len(missing_obs) > 0:
                        cheat_score += 1
            except ValueError as e:
                logger.warning(f"Something went wrong parsing the target for query {query}: {e}")
            except Exception as e:
                logger.error(f"Something unexpected went wrong parsing the target for query {query}: {e}. "
                             f"Please fix :(")

            prompt = self.generate_prompt(query, ranked_nodes)

            input_sequences.append(prompt)
            output_sequences.append(target)

        return input_sequences, output_sequences, cheat_score

    def _add_special_tokens(self, input_ids: list[list], max_length: int) -> list[int]:
        """Add special tokens and right aligned padding to tokenized input_ids"""
        new_ids = [
            [self.tokenizer.cls_token_id] +
            x[:min(len(x), max_length - 2)] +
            [self.tokenizer.eos_token_id if isinstance(self.tokenizer, RobertaTokenizerFast)
             else self.tokenizer.sep_token_id] +
            [self.tokenizer.pad_token_id] * (max_length - min(len(x) + 2, max_length))
            for x in input_ids
        ]

        return new_ids

    @staticmethod
    def _add_special_tokens_mask(attention_mask: list[list], max_length: int) -> list[int]:
        """Add special tokens and right aligned padding to tokenized attention_mask"""
        new_mask = [
            [1] + x[:min(len(x), max_length - 2)] + [1] +
            [0] * (max_length - min(len(x) + 2, max_length))
            for x in attention_mask
        ]
        return new_mask

    def encode_data(self, data: pd.DataFrame, file_path: str) -> dict:
        """
            Encode given CSV data to create train or test dataset to train
            and evaluate model on. Uses multiprocessing.

            :param data: DataFrame containing query-sexp pairs
            :param file_path: path to pickle encoded dataset to
            :returns: encoded data as dictionary containing
                      {input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels}
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                return pickle.load(file)

        # input_sequences, output_sequences = self.generate_input_sequences(data, 0)

        proc_split = np.array_split(data, min(len(data), 4))
        pool = multiprocessing.Pool()
        input_sequences, output_sequences, cheat_score = \
            zip(*pool.starmap(self.generate_input_sequences, [(x, i) for i, x in enumerate(proc_split)]))
        pool.close()
        pool.join()

        input_sequences = list(chain(*input_sequences))
        output_sequences = list(chain(*output_sequences))

        logger.info(f"Done generating prompts. Number of 'cheated' prompts: {sum(cheat_score)}")

        # Tokenize prompt and pad/truncate where needed
        encodings = self.tokenizer(input_sequences, padding=False, add_special_tokens=False,
                                   max_length=self.max_source_length, truncation=True)

        # Add correct special tokens and do right-aligned padding
        input_ids = torch.tensor(self._add_special_tokens(encodings.input_ids, max_length=self.max_source_length))
        attention_mask = torch.tensor(
            self._add_special_tokens_mask(encodings.attention_mask, max_length=self.max_source_length)
        )

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        if len(output_sequences):
            # TODO: attempt a custom tokenizer with limited vocabulary
            target_encoding = self.tokenizer(output_sequences, padding=False, add_special_tokens=False,
                                             max_length=self.max_target_length, truncation=True)

            decoder_input_ids = self._add_special_tokens(target_encoding.input_ids, max_length=self.max_target_length)

            # TODO: let Huggingface handle right-shift of labels (https://discuss.huggingface.co/t/could-i-inference-the-encoder-decoder-model-without-specify-decoder-input-ids/5811/3)
            # res['decoder_input_ids'] = torch.tensor(decoder_input_ids)
            # res['decoder_attention_mask'] = torch.tensor(
            #     self._add_special_tokens_mask(target_encoding.attention_mask, max_length=self.max_target_length)
            # )

            # replace padding token id's of the labels by -100
            labels = torch.tensor(decoder_input_ids)
            labels[labels == self.tokenizer.pad_token_id] = -100
            res['labels'] = labels

        with open(file_path, 'wb') as file:
            pickle.dump(res, file)

        return res

    def train(self, data_path, output_dir):
        output_dir = output_dir.rstrip('/')

        df = pd.read_csv(data_path, delimiter=';')
        train_data, test_data = train_test_split(df, test_size=.1, random_state=self.seed)

        train_encodings = self.encode_data(train_data, paths_config.TRAIN_DATA_PATH)
        test_encodings = self.encode_data(test_data, paths_config.TEST_DATA_PATH)

        train_dataloader = DataLoader(HFDataset(train_encodings), shuffle=True, batch_size=self.batch_size)
        test_dataloader = DataLoader(HFDataset(test_encodings), batch_size=self.batch_size)

        accelerator = Accelerator(project_dir=output_dir)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)  # lr=5e-4, eps=1e-8
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps,
                                                    num_training_steps=self.epochs * len(train_dataloader))
        self.device = accelerator.device

        # Put all settings and tensors to correct device
        self.model, optimizer, train_dataloader, test_dataloader = accelerator.prepare([
            self.model, optimizer, train_dataloader, test_dataloader
        ])

        start_epoch = 0
        checkpoint_step = 0
        if self.checkpoint is not None:
            dir_ = self.checkpoint.split('/')[-1]
            if not (epoch_step := re.match(r"\d{1,3}_\d{1,5}", dir_)):
                logger.error(f"Can't infer last epoch and step from checkpoint path {self.checkpoint}."
                             f"Make sure the checkpoint folder ends with pattern `\\d{{1,3}}_\\d{{1,5}}$`.")

            # Restore state of the model and dataloader based on the last checkpoint
            logger.info(f"Loading model checkpoint to continue training from {self.checkpoint}")
            accelerator.load_state(self.checkpoint)
            start_epoch = int(epoch_step[0].split('_')[0])
            checkpoint_step = int(epoch_step[0].split('_')[-1])
            accelerator.skip_first_batches(train_dataloader, checkpoint_step % len(train_dataloader))
        else:
            accelerator.save_state(f"{output_dir}/checkpoints/0_0")  # Save initial state

        running_loss = 0
        eval_loss = None
        best_loss = float("inf")
        metrics = {'rouge2': None, 'bleu': None}
        METRICS_LOG_FILE = f"{output_dir}/training_metrics_log_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        with open(METRICS_LOG_FILE, 'w', newline='') as logfile:
            writer = csv.writer(logfile)
            writer.writerow(['epoch', 'step', 'loss/train', 'loss/eval', 'rouge2', 'bleu'])

        pbar = tqdm(range(self.epochs * len(train_dataloader)),
                    desc=f"*Ppprt prrt* Training model...".zfill(3),
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')
        pbar.update(start_epoch * len(train_dataloader) + checkpoint_step)
        for epoch in range(start_epoch, self.epochs):
            for step, batch in enumerate(train_dataloader):
                self.model.train()

                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                loss = outputs.loss
                running_loss += loss  # TODO: not used atm.
                accelerator.backward(loss)

                if step % self.gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    eval_line = {'loss/train': loss.item(), 'loss/eval': eval_loss} | metrics
                    pbar.set_postfix(eval_line)

                    with open(METRICS_LOG_FILE, 'a', newline='') as logfile:
                        writer = csv.DictWriter(logfile,
                                                fieldnames=['epoch', 'step'] + list(eval_line.keys()),
                                                quoting=csv.QUOTE_MINIMAL)
                        writer.writerow({'epoch': epoch, 'step': step} | eval_line)

                if step % self.eval_steps == 0 and step != 0:
                    eval_loss, metrics = self.eval(test_dataloader, accelerator)

                    accelerator.save_state(f"{output_dir}/checkpoints/{epoch}_{step}")  # save checkpoint
                    accelerator.wait_for_everyone()

                    if eval_loss < best_loss:
                        unwrapped_model = accelerator.unwrap_model(self.model)
                        unwrapped_model.save_pretrained(f"{output_dir}/model", save_function=accelerator.save)
                        best_loss = eval_loss

                pbar.update(1)
        pbar.close()

    def eval(self, dataloader: DataLoader, accelerator: Accelerator) -> (float, dict):
        metrics = {}
        losses = []
        self.model.eval()
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

            batch_metrics = self.compute_metrics(outputs.logits, batch['labels'])
            metrics = {m: metrics.get(m, []) + [v] for m, v in batch_metrics.items()}
            losses.append(accelerator.gather(outputs.loss))

        loss = torch.mean(torch.stack(losses))
        return loss.item(), {m: np.mean(v) for m, v in metrics.items()}

    def compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        pred_ids = torch.argmax(logits, axis=-1)

        # all unnecessary tokens are removed
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels[labels == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge = self.rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"]
        bleu = self.bleu.compute(predictions=pred_str, references=label_str)["score"]
        return {"rouge2": round(rouge, 4), "bleu": round(bleu, 4)}

    def conditional_generation(self, query: str, verbose: bool = False) -> (SExpression, OrderedDict):
        """
            Inference function to generate an S-expression based on a given query
            using constrained beam search. Returns the best scoring generated valid
            S-expression and the ranked candidate nodes the expression is based on.

            :param query: query string to inference
            :param verbose: print status updates in console
            :returns: tuple containing a valid S-expression and the ranked MSR/DIM nodes it is based on
        """
        ranked_nodes, subgraph = \
            self.entity_retriever.get_candidate_nodes_embeddings(query, full_graph=True, verbose=verbose)

        if len(ranked_nodes) == 0:
            return None, {}  # TODO: handle properly

        prompt = self.generate_prompt(query, ranked_nodes)
        encodings = self.tokenizer([prompt], padding=False, add_special_tokens=False,
                                   max_length=self.max_source_length, truncation=True)
        input_ids = torch.tensor(self._add_special_tokens(encodings.input_ids, max_length=self.max_source_length))
        attention_mask = torch.tensor(
            self._add_special_tokens_mask(encodings.attention_mask, max_length=self.max_source_length)
        )

        space_token = (self.tokenizer.tokenize(' ', add_special_tokens=False) or [' '])[0]
        special_tokens = set(self.tokenizer.special_tokens_map.values())
        sexp_beam_cache: Dict[int, SExpression] = {hash(''): SExpression(subgraph)}

        def _hash_beam(beam: list[str]) -> int:
            str_hash = ' '.join([t for t in beam
                                 if t not in {space_token, SOS, f"{space_token}{SOS}"} | special_tokens])
            return hash(str_hash)

        def _restrict_decode_vocab(batch_id, prefix_beam) -> list[int]:
            """
                Helper function to get admissible tokens during beam search for generating the S-expression.

                :param batch_id: not used
                :param prefix_beam: generated beam so far
                :returns list containing token ids of admissible tokens
            """
            beam_str = self.tokenizer.convert_ids_to_tokens(prefix_beam)
            str_hash = _hash_beam(beam_str)
            if str_hash not in sexp_beam_cache:
                sexp_beam_cache[str_hash] = deepcopy(sexp_beam_cache[_hash_beam(beam_str[:-1])])
                sexp_beam_cache[str_hash].add_token(beam_str[-1])

            sexp = sexp_beam_cache[str_hash]
            admissible_tokens = sexp.get_admissible_tokens()
            if len(admissible_tokens) == 0:
                return [self.tokenizer.pad_token_id]

            if set(admissible_tokens) == {WHERE.__name__, EOS} and beam_str[-1] == f"{space_token}{SOS}":
                # Continue with WHERE clause if beam already chose to open a statement
                return self.tokenizer(WHERE.__name__, add_special_tokens=False)['input_ids']

            if OR.__name__ in admissible_tokens and beam_str[-1] == f"{space_token}{SOS}":
                # Continue with OR clause if beam already chose to open a statement
                return self.tokenizer(OR.__name__, add_special_tokens=False)['input_ids']

            if (EOS in admissible_tokens and len(admissible_tokens) > 1
                    and beam_str[-1] in [space_token, SOS, f"{space_token}{SOS}"]):
                # Continue with code node if beam already chose to add a space in an OR statement
                admissible_tokens.remove(EOS)

            # TODO: make sure this works for word pieces in case a admissible token is not in the tokenizer vocabulary

            beam_vocab = set()
            for token in admissible_tokens:
                type_ = getattr(logical_forms, token, None)

                if type_ is None and issubclass(getattr(logical_forms, beam_str[-1], type(None)), AGGREGATION):
                    # special case when adding TABLE node; should have an SOS before
                    beam_vocab.add(f"{'' if len(beam_str) <= 1 else ' '}{SOS}")
                    continue

                if ((type_ is None or issubclass(type_, TerminalNode))
                        and beam_str[-1] not in [space_token, SOS, f"{space_token}{SOS}"]
                        and token != EOS
                        and space_token in self.tokenizer.vocab):
                    beam_vocab.add(' ')
                    continue

                if (type_ is not None and not issubclass(type_, TerminalNode)
                        and beam_str[-1] not in [SOS, f"{space_token}{SOS}"]):
                    beam_vocab.add(f"{'' if len(beam_str) <= 1 else ' '}{SOS}")
                    continue

                beam_vocab.add(token)

            tok_ids = self.tokenizer(list(beam_vocab), add_special_tokens=False)['input_ids']
            return list(chain(*tok_ids))

        stopping_criteria = StoppingCriteriaList([
            SExpressionStoppingCriteria(tokenizer=self.tokenizer)
        ])

        outputs = self.model.generate(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_length=self.max_target_length,
            do_sample=True,
            # top_k=0,  # TODO: experiment with these values
            # temperature=0.7,
            # top_p=0.92,
            stopping_criteria=stopping_criteria,
            early_stopping=False,  # Stopping criterion is controlled by S-expression restrictions
            decoder_start_token_id=0 if isinstance(self.tokenizer, RobertaTokenizerFast) else self.tokenizer.cls_token_id,
            bad_words_ids=[[0]],
            num_beams=self.beam_size,  # TODO: determine good speed-performance balance
            num_return_sequences=self.beam_size,
            output_scores=True,
            return_dict_in_generate=True,
            prefix_allowed_tokens_fn=_restrict_decode_vocab
        )

        sequences = list(zip(
            self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True),
            outputs.sequences_scores
        ))

        sequences.sort(key=lambda x: x[1], reverse=True)
        for expression, score in sequences:
            try:
                sexp = SExpression.parse(expression, graph=subgraph)
                return sexp, ranked_nodes.get(str(sexp._current_table.uri), [None, {}])[1]
            except AssertionError as e:
                # This should never happen
                logger.error(f"Generated a non-parsable S-expression. This should not be possible: {e}")
                continue

        raise ValueError("Failed to generate any valid S-expression")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(paths_config.TOKENIZER_EXT_PATH)
    encoder = paths_config.START_MODEL_PATH
    decoder = paths_config.START_MODEL_PATH

    seq2seq = SExpressionEncoderDecoderTrainer(tokenizer, encoder, decoder,
                                               eval_steps=200,
                                               batch_size=4,
                                               learning_rate=3e-5,
                                               special_tokens_weights_factor=1e-2,
                                               faiss_solution=3)
    seq2seq.train(data_path=paths_config.DATASET_PATH,
                  output_dir=paths_config.TRAINED_MODEL_PATH)
