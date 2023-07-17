import faiss
import logging
import torch
from faiss import IndexIDMap2
from transformers import (
    PreTrainedModel, PretrainedConfig, EncoderDecoderModel,
    RobertaPreTrainedModel, EncoderDecoderConfig
)
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, create_position_ids_from_input_ids
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from typing import Optional, Union, Tuple

import paths_config

logger = logging.getLogger(__name__)


class SExpressionEmbeddings(RobertaEmbeddings):
    def __init__(self, config, faiss_index: IndexIDMap2, faiss_solution: int):
        super().__init__(config)

        self.faiss_index = faiss_index
        self.faiss_solution = faiss_solution

    def initialize_faiss_weights(self):
        """
            SOLUTION 3:
            Use the same embedding layer as usual, but override the embedding matrix to accommodate
            for the new tokenizer vocab size and initialize the vectors with the vectors found in the
            Faiss-index.

            https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512
        """
        assert self.faiss_solution == 3

        faiss_ids = faiss.vector_to_array(self.faiss_index.id_map)
        if max(faiss_ids) > self.word_embeddings.weight.shape[0]:
            raise ValueError("Weight matrix of embedding layer is smaller than the largest ID in the"
                             "tokenizer vocabulary and Faiss-index. Call `model.{encoder, decoder}.resize."
                             "resize_token_embeddings(len(self.tokenizer))` first before settings the"
                             "weights of the newly randomly initialized embeddings.")

        weights = self.faiss_index.reconstruct_batch(faiss_ids)
        with torch.no_grad():
            self.word_embeddings.weight[faiss_ids] = torch.tensor(weights).to(self.word_embeddings.weight.device)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None, past_key_values_length=0):
        """
            SOLUTION 2:
            Use a custom embedding layer and override the relevant code nodes with vectors
            obtained from Faiss when doing the forward loop.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

            if self.faiss_solution == 2:
                # TODO: use batch reconstruction
                faiss_ids = faiss.vector_to_array(self.faiss_index.id_map)
                for idx, input_id in enumerate(input_ids[0]):
                    if input_id.item() in faiss_ids:
                        # Override embeddings for the code which have a contextual embedding in the pre-computed index
                        inputs_embeds[0][idx] = torch.tensor(self.faiss_index.reconstruct(input_id.item()))

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SExpressionEncoderDecoder(EncoderDecoderModel):
    """
        Mostly based on EncoderDecoderModel
    """
    def __init__(
            self,
            config: Optional[PretrainedConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            special_tokens_weights: Optional[torch.Tensor] = None,
            label_smoothing: Optional[float] = None
    ):
        super().__init__(config, encoder, decoder)

        loss_weights = (special_tokens_weights if special_tokens_weights is not None
                        else getattr(config, 'special_tokens_weights', None))
        if len(loss_weights) != self.encoder.get_input_embeddings().num_embeddings:
            logger.warning(f"Size mismatch between model embedding shape "
                           f"({self.encoder.get_input_embeddings().num_embeddings}) and number of loss weights "
                           f"({len(loss_weights)}). Setting loss_fct.weight to None. Make sure to re-initialize"
                           f"the embedding matrix before and (re)setting the loss weights.")
            loss_weights = None

        loss_smoothing = (label_smoothing if label_smoothing is not None
                          else getattr(config, 'label_smoothing', None))
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=loss_smoothing)

        if hasattr(config, 'special_tokens_weights'):
            del config.special_tokens_weights  # remove tensor as is doesn't serialize to config.json when saving model

        self.faiss_index = faiss.read_index(paths_config.FAISS_PATH)
        self.faiss_solution = config.faiss_solution

        # SOLUTION 2 & 3
        if self.faiss_solution != 1:
            embedding_layer = SExpressionEmbeddings(self.config.encoder, self.faiss_index,
                                                    faiss_solution=self.faiss_solution)
            # Freeze embedding layer
            for param in embedding_layer.parameters():
                param.requires_grad = False

            self.encoder.embeddings = embedding_layer
            if isinstance(self.decoder, RobertaPreTrainedModel):
                self.decoder.roberta.embeddings = embedding_layer
            else:
                self.decoder.bert.embeddings = embedding_layer

    def embed(self, embedder: RobertaEmbeddings, input_ids: torch.LongTensor):
        """
            SOLUTION 1:
            Use default embedding layer of encoder and decoder. Generate word embedding
            for the entire input sequence and override candidate node codes with pre-computed
            contextual embeddings from the Faiss-index.

            :param embedder: embeddings layer for the specific encoder or decoder.
                             self.encoder.embeddings or self.decoder.roberta.embeddings
            :param input_ids:
        """
        # Generate word embedding for the entire sequence
        inputs_embeds = embedder.word_embeddings(input_ids)
        faiss_ids = faiss.vector_to_array(self.faiss_index.id_map)
        for idx, input_id in enumerate(input_ids[0]):
            if input_id.item() in faiss_ids:
                # Override embeddings for the code which have a contextual embedding in the pre-computed index
                inputs_embeds[0][idx] = torch.tensor(self.faiss_index.reconstruct(input_id.item()))

        # Generate positional embeddings
        embedding_output = embedder(
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        return embedding_output

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
            past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if self.faiss_solution == 1:
                # Use custom embedder to embed candidate node codes in input
                inputs_embeds = self.embed(self.encoder.embeddings, input_ids)  # SOLUTION 1

            encoder_outputs = self.encoder(
                input_ids=input_ids if self.faiss_solution != 1 else None,  # SOLUTION 2 & 3
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if (labels is not None) and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if self.faiss_solution == 1:
            # Use custom embedder to embed target node codes in output
            if isinstance(self.decoder, RobertaPreTrainedModel):
                embeddings = self.decoder.roberta.embeddings
            else:
                embeddings = self.decoder.bert.embeddings
            decoder_inputs_embeds = self.embed(embeddings, decoder_input_ids)  # SOLUTION 1

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids if self.faiss_solution != 1 else None,  # SOLUTION 2 & 3
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent of decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss = self.loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
