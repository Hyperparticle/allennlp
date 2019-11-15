"""
Decodes sequences of tags, e.g., POS tags, given a list of contextualized word embeddings
"""

from typing import Optional, Dict, List
from overrides import overrides

import numpy
import torch

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.data.tokenizers import lemma_util
from allennlp.models import TaggerMultitask


def _decode_lemma(word, rule):
    if rule == "_":
        # Any rule with "_" indicates that lemmas are null, so return null as well
        return "_"
    if rule == "@@UNKNOWN@@":
        # Sometimes the lemmatizer might not know the right rule, so apply the identity function
        return word
    return lemma_util.apply_lemma_rule(word, rule)


@Model.register("lemmatizer_multitask")
class LemmatizerMultitask(TaggerMultitask):
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
        super().__init__(
            vocab,
            encoder,
            calculate_span_f1,
            label_encoding,
            label_namespace,
            verbose_metrics,
            label_smoothing,
            dropout,
            adaptive,
            adaptive_class_sizes,
            initializer,
            regularizer
        )

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_words = output_dict["words"]

        all_predictions = output_dict["class_probabilities"]
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions, words in zip(predictions_list, all_words):
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace=self.label_namespace)
                    for x in argmax_indices]
            tags = [_decode_lemma(word, rule) for word, rule in zip(words, tags)]
            all_tags.append(tags)
        output_dict["tags"] = all_tags

        return output_dict
