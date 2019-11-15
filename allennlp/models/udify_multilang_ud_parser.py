"""
The base UDify model for training and prediction
"""

from typing import Optional, Any, Dict, List, Tuple
from overrides import overrides
import logging

from collections import defaultdict
import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, dynamic_mask
from allennlp.training.metrics import AttachmentScores, CategoricalAccuracy

logger = logging.getLogger(__name__)


@Model.register("udify_multilang_ud_parser")
class UDifyMultilangUDParser(Model):
    """
    The UDify model base class. Applies a sequence of shared encoders before decoding in a multi-task configuration.
    Uses TagDecoder and DependencyDecoder to decode each UD task.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 tasks: List[str],
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoders: Dict[str, Model],
                 dropout: float = 0.0,
                 word_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(UDifyMultilangUDParser, self).__init__(vocab, regularizer)

        self.tasks = tasks
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained("bert-base-multilingual-cased").vocab
        self.text_field_embedder = text_field_embedder
        self.shared_encoder = encoder
        self.word_dropout = word_dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        self.decoders = torch.nn.ModuleDict(decoders)

        self.metrics = {}
        self.lang_acc: Dict[(str, str), CategoricalAccuracy] = defaultdict(CategoricalAccuracy)
        self.lang_attachment: Dict[str, AttachmentScores] = defaultdict(AttachmentScores)

        for task in self.tasks:
            if task not in self.decoders:
                raise ConfigurationError(f"Task {task} has no corresponding decoder. Make sure their names match.")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        initializer(self)

    @overrides
    def forward(self,
                words: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                **kwargs: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        gold_tags = kwargs

        batch_lang = metadata[0].get("lang", None) if metadata else None
        if batch_lang:
            for entry in metadata:
                if entry["lang"] != batch_lang:
                    logger.warn("Two languages in the same batch: {}, {}".format(entry["lang"], batch_lang))

        mask = get_text_field_mask(words)
        self._apply_token_dropout(words)

        embedded_text_input = self.text_field_embedder(words)
        encoded_text = self.shared_encoder(embedded_text_input, mask)

        output_dict = {}

        # Run through each of the tasks on the shared encoder and save predictions
        for task in self.tasks:
            # if self.scalar_mix:
            #     decoder_input = self.scalar_mix[task](encoded_text, mask)

            if task == "deps":
                pred_output = self._dependency_parse(encoded_text, mask, gold_tags, metadata, batch_lang)
            else:
                pred_output = self._tagger_parse(task, encoded_text, mask, gold_tags, metadata, batch_lang)

            output_dict[task] = pred_output

            # if f"{task}_tags" in gold_tags or task == "deps" and "head_tags" in gold_tags:
            #     # Keep track of the loss if we have the gold tags available
            #     losses[task] = pred_output["task"]["loss"]

        if gold_tags:
            output_dict["loss"] = sum((output_dict[task]["loss"] for task in self.tasks), 0)

        if metadata is not None:
            for tag in ["lang", "words", "ids", "multiword_ids", "multiword_forms"]:
                output_dict[tag] = [x[tag] for x in metadata if tag in x]

        return output_dict

    def _dependency_parse(self, decoder_input, mask, gold_tags, metadata, batch_lang):
        decoder = self.decoders["deps"]

        pos_tags = gold_tags.get("pos_tags", None)
        head_tags = gold_tags.get("head_tags", None)
        head_indices = gold_tags.get("head_indices", None)
        pred_output = decoder(decoder_input,
                              mask,
                              head_tags,
                              head_indices,
                              pos_tags,
                              metadata)

        if batch_lang:
            batch_size = decoder_input.size(0)
            score_mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
            evaluation_mask = (decoder._get_mask_for_eval(score_mask[:, 1:], pos_tags)
                               if pos_tags is not None
                               else score_mask[:, 1:])
            self.lang_attachment[batch_lang](
                pred_output["heads"][:, 1:],
                pred_output["head_tags"][:, 1:],
                head_indices,
                head_tags,
                evaluation_mask
            )
        return pred_output

    def _tagger_parse(self, task, decoder_input, mask, gold_tags, metadata, batch_lang):
        decoder = self.decoders[task]

        tags = gold_tags.get(f"{task}_tags", None)
        pred_output = decoder(decoder_input, mask, tags, metadata)

        if batch_lang:
            self.lang_acc[(batch_lang, task)](pred_output["class_probabilities"], tags, mask.float())

        return pred_output

    def _apply_token_dropout(self, tokens):
        if "bert" in tokens and self.training:
            oov_token = self.bert_vocab["[MASK]"]
            ignore_tokens = [self.bert_vocab[token] for token in ["[PAD]", "[CLS]", "[SEP]"]]
            tokens["bert"] = dynamic_mask(tokens["bert"],
                                          oov_token=oov_token,
                                          padding_tokens=ignore_tokens,
                                          prob=self.word_dropout)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for task in self.tasks:
            decoder = self.decoders[task]
            output = decoder.decode(output_dict[task])
            for k, v in output.items():
                output_dict[f"{task}.{k}"] = v

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_track = {"upos", "xpos", "feats", "lemmas", "LAS", "UAS"}

        metrics = {f".run/{task}/{name}": task_metric
                   for task in self.tasks
                   for name, task_metric in self.decoders[task].get_metrics(reset).items()
                   if metrics_to_track.intersection({task, name})}

        # The "avg" metric is a global measure for early stopping and saving
        metrics[".run/_avg"] = sum(metrics.values()) / len(metrics)

        for (lang, task), metric in self.lang_acc.items():
            metrics[f"_{task}/{lang}"] = metric.get_metric(reset)

        for lang, scores in self.lang_attachment.items():
            lang_metrics = scores.get_metric(reset)
            for task in ["UAS", "LAS"]:
                metrics[f"_{task}/{lang}"] = lang_metrics[task]

        return metrics
