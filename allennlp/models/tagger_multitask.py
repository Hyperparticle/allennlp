"""
Decodes sequences of tags, e.g., POS tags, given a list of contextualized word embeddings
"""

from typing import Optional, Any, Dict, List
from overrides import overrides
import logging

import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.adaptive import AdaptiveLogSoftmaxWithLoss

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits, sequence_cross_entropy
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PassThroughTokenEmbedder

logger = logging.getLogger(__name__)


@Model.register("tagger_multitask")
class TaggerMultitask(SimpleTagger):
    """
    A basic sequence tagger that decodes from inputs of word embeddings
    """
    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2SeqEncoder,
        calculate_span_f1: bool = None,
        label_encoding: Optional[str] = None,
        label_namespace: str = "labels",
        verbose_metrics: bool = False,
        label_smoothing: float = 0.0,
        dropout: float = 0.0,
        adaptive: bool = False,
        adaptive_class_sizes: List[float] = (0.05, 0.2),
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:
        # We won't use the text field embedder, so pass a dummy embedder with the correct output dim
        text_field_embedder = BasicTextFieldEmbedder({"": PassThroughTokenEmbedder(encoder.get_output_dim())})

        super().__init__(
            vocab,
            text_field_embedder,
            encoder,
            calculate_span_f1,
            label_encoding,
            label_namespace,
            verbose_metrics,
            label_smoothing,
            dropout,
            initializer,
            regularizer
        )

        self.adaptive = adaptive

        if self.adaptive:
            adaptive_cutoffs = [int(round(self.num_classes * class_size)) for class_size in adaptive_class_sizes]
            logger.info("Using adaptive softmax with cutoffs ({}): {}".format(label_namespace, adaptive_cutoffs))
            self.tag_projection_layer = AdaptiveLogSoftmaxWithLoss(encoder.get_output_dim(),
                                                                   self.num_classes,
                                                                   cutoffs=adaptive_cutoffs)
        else:
            self.tag_projection_layer = TimeDistributed(
                Linear(self.encoder.get_output_dim(), self.num_classes)
            )

        self.metrics = {
            "acc": CategoricalAccuracy(),
        }

    @overrides
    def forward(
        self,  # type: ignore
        embedded_text_input: torch.FloatTensor,
        mask: torch.LongTensor,
        tags: torch.LongTensor,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, _ = embedded_text_input.size()
        encoded_text = self.encoder(embedded_text_input, mask)
        output_dim = [batch_size, sequence_length, self.num_classes]

        loss_fn = self._adaptive_loss if self.adaptive else self._loss
        output_dict = loss_fn(encoded_text, mask, tags, output_dim)

        if tags is not None:
            logits = output_dict["logits"]

            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            if self._f1_metric is not None:
                self._f1_metric(logits, tags, mask.float())

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    def _loss(self, encoded_text, mask, tags, output_dim):
        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(output_dim)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if tags is not None:
            output_dict["loss"] = sequence_cross_entropy_with_logits(logits,
                                                                     tags,
                                                                     mask,
                                                                     label_smoothing=self.label_smoothing)
        return output_dict

    def _adaptive_loss(self, encoded_text, mask, tags, output_dim):
        flat_outputs = encoded_text.view(-1, encoded_text.size(-1))
        class_probabilities = self.tag_projection_layer.log_prob(flat_outputs).view(output_dim)

        output_dict = {"logits": class_probabilities, "class_probabilities": class_probabilities}

        if tags is not None:
            output_dict["loss"] = sequence_cross_entropy(class_probabilities,
                                                         tags,
                                                         mask,
                                                         label_smoothing=self.label_smoothing)
        return output_dict
