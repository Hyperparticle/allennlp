"""
Decodes dependency trees given a list of contextualized word embeddings
"""

from typing import Dict, Optional, Any, List
import logging

import torch

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder
from allennlp.models.biaffine_dependency_parser import BiaffineDependencyParser
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PassThroughTokenEmbedder


logger = logging.getLogger(__name__)


@Model.register("biaffine_parser_multitask")
class DependencyDecoder(BiaffineDependencyParser):
    """
    Modifies BiaffineDependencyParser, removing the input TextFieldEmbedder dependency to allow the model to
    essentially act as a decoder when given intermediate word embeddings instead of as a standalone model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2SeqEncoder,
        tag_representation_dim: int,
        arc_representation_dim: int,
        tag_feedforward: FeedForward = None,
        arc_feedforward: FeedForward = None,
        use_mst_decoding_for_validation: bool = True,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:
        # We won't use the text field embedder, so pass a dummy embedder with the correct output dim
        text_field_embedder = BasicTextFieldEmbedder({"": PassThroughTokenEmbedder(encoder.get_output_dim())})

        super().__init__(
            vocab,
            text_field_embedder,
            encoder,
            tag_representation_dim,
            arc_representation_dim,
            tag_feedforward,
            arc_feedforward,
            None,  # pos_tag_embedding
            use_mst_decoding_for_validation,
            dropout,
            input_dropout,
            initializer,
            regularizer,
        )

    def forward(
        self,  # type: ignore
        # words: Dict[str, torch.LongTensor],
        embedded_text_input: torch.FloatTensor,
        mask: torch.LongTensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        pos_tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll = self._parse(
            embedded_text_input, mask, head_tags, head_indices
        )

        loss = arc_nll + tag_nll

        if head_indices is not None and head_tags is not None:
            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags) if pos_tags is not None else mask[:, 1:]
            # We calculate attatchment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(
                predicted_heads[:, 1:],
                predicted_head_tags[:, 1:],
                head_indices,
                head_tags,
                evaluation_mask,
            )

        output_dict = {
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "arc_loss": arc_nll,
            "tag_loss": tag_nll,
            "loss": loss,
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
        }

        return output_dict
