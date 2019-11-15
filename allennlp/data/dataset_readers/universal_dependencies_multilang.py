from typing import Dict, Tuple, List, Iterator, Any, Callable
import logging
import itertools
import glob
import os
import numpy as np

from overrides import overrides
from conllu import parse_incr

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, lemma_util

logger = logging.getLogger(__name__)


def get_file_paths(pathname: str, languages: List[str] = None):
    """
    Gets a list of all files by the pathname with the given language ids.
    Filenames are assumed to have the language identifier followed by a dash
    as a prefix (e.g. en-universal.conll).

    Parameters
    ----------
    pathname :  ``str``, required.
        An absolute or relative pathname (can contain shell-style wildcards)
    languages : ``List[str]``, optional (default = None)
        The language identifiers to use. Set to None to use all available languages.

    Returns
    -------
    A list of tuples (language id, file path).
    """
    paths = []
    for file_path in sorted(glob.glob(pathname)):
        base = str(os.path.splitext(os.path.basename(file_path))[0])
        lang_id = base.split("-")[0]
        if languages is None or lang_id in languages:
            paths.append((lang_id, file_path))

    if not paths:
        raise ConfigurationError("No dataset files to read")

    return paths


@DatasetReader.register("universal_dependencies_multilang")
class UniversalDependenciesMultiLangDatasetReader(DatasetReader):
    """
    Reads multiple files in the conllu Universal Dependencies format.
    All files should be in the same directory and the filenames should have
    the language identifier followed by a dash as a prefix (e.g. en-universal.conll)
    When using the alternate option, the reader alternates randomly between
    the files every instances_per_file. The is_first_pass_for_vocab disables
    this behaviour for the first pass (could be useful for a single full path
    over the dataset in order to generate a vocabulary).

    Notice: when using the alternate option, one should also use the ``instances_per_epoch``
    option for the iterator. Otherwise, each epoch will loop infinitely.

    Parameters
    ----------
    languages : ``List[str]``, optional (default = None)
        The language identifiers to use. Set to None to use all available languages.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    alternate : ``bool``, optional (default = True)
        Whether to alternate between input files.
    is_first_pass_for_vocab : ``bool``, optional (default = True)
        Whether the first pass will be for generating the vocab. If true,
        the first pass will run over the entire dataset of each file (even if alternate is on).
    instances_per_file : ``int``, optional (default = 32)
        The amount of consecutive cases to sample from each input file when alternating.
    """

    def __init__(
        self,
        languages: List[str] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        lazy: bool = False,
        alternate: bool = True,
        is_first_pass_for_vocab: bool = True,
        instances_per_file: int = 32,
    ) -> None:
        super().__init__(lazy)
        self._languages = languages
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._use_language_specific_pos = use_language_specific_pos

        self._is_first_pass_for_vocab = is_first_pass_for_vocab
        self._alternate = alternate
        self._instances_per_file = instances_per_file

        self._is_first_pass = True
        self._iterators: List[Tuple[str, Iterator[Any]]] = None

    def _read_one_file(self, lang: str, file_path: str):
        with open(file_path, "r") as conllu_file:
            logger.info(
                "Reading UD instances for %s language from conllu dataset at: %s", lang, file_path
            )

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if isinstance(x["id"], int)]
                multiword_tokens = [x for x in annotation if isinstance(x["id"], tuple)]

                if not annotation:
                    # Empty lines should be skipped
                    continue

                def get_field(field: str,
                              map_fn: Callable[[Any], Any] = None,
                              sentence: List[Any] = tuple(annotation)) -> List[Any]:
                    """
                    Calls the function on the annotation with the given field name.
                    Returns '_' if the field is empty (conllu's way of indicating null).
                    """
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [map_fn(line[field]) if line[field] is not None else "_"
                            for line in sentence if field in line]

                ids = get_field("id")
                multiword_ids = get_field("multi_id", sentence=multiword_tokens)
                multiword_forms = get_field("form", sentence=multiword_tokens)

                words = get_field("form")
                lemmas = get_field("lemma")
                lemmas_tags = [lemma_util.gen_lemma_rule(word, lemma)
                               if lemma != "_" else "_"
                               for word, lemma in zip(words, lemmas)]
                upos_tags = get_field("upostag")
                xpos_tags = get_field("xpostag")
                pos_tags = xpos_tags if self._use_language_specific_pos else upos_tags
                feats_tags = get_field("feats", lambda x: "|".join(k + "=" + v for k, v in x.items())
                                                          if hasattr(x, "items") else "_")
                heads = get_field("head")
                dep_rels = get_field("deprel")
                dependencies = list(zip(dep_rels, heads))

                yield self.text_to_instance(lang, words, lemmas, upos_tags, xpos_tags, pos_tags, feats_tags,
                                            lemmas_tags, dependencies, ids, multiword_ids, multiword_forms)

    @overrides
    def _read(self, file_path: str):
        file_paths = get_file_paths(file_path, self._languages)
        if (self._is_first_pass and self._is_first_pass_for_vocab) or (not self._alternate):
            iterators = [
                iter(self._read_one_file(lang, file_path)) for (lang, file_path) in file_paths
            ]
            self._is_first_pass = False
            for inst in itertools.chain(*iterators):
                yield inst

        else:
            if self._iterators is None:
                self._iterators = [
                    (lang, iter(self._read_one_file(lang, file_path)))
                    for (lang, file_path) in file_paths
                ]
            num_files = len(file_paths)
            while True:
                ind = np.random.randint(num_files)
                lang, lang_iter = self._iterators[ind]
                for _ in range(self._instances_per_file):
                    try:
                        yield lang_iter.__next__()
                    except StopIteration:
                        lang, file_path = file_paths[ind]
                        lang_iter = iter(self._read_one_file(lang, file_path))
                        self._iterators[ind] = (lang, lang_iter)
                        yield lang_iter.__next__()

    @overrides
    def text_to_instance(
        self,  # type: ignore
        lang: str,
        words: List[str],
        lemmas: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        pos_tags: List[str] = None,
        feats_tags: List[str] = None,
        lemmas_tags: List[str] = None,
        dependencies: List[Tuple[str, int]] = None,
        ids: List[str] = None,
        multiword_ids: List[str] = None,
        multiword_forms: List[str] = None
    ) -> Instance:

        """
        Parameters
        ----------
        lang : ``str``, required.
            The language identifier.
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        lemmas: ``List[str]``, optional (default = None)
            The universal dependencies lemmas for each word.
        upos_tags : ``List[str]``, optional (default = None)
            The universal dependencies UPOS tags for each word.
        xpos_tags : ``List[str]``, optional (default = None)
            The universal dependencies XPOS tags for each word.
        pos_tags : ``List[str]``, optional (default = None)
            The POS tags for each word given by self.use_language_specific_pos.
        feats_tags: ``List[str]``, optional (default = None)
            The universal dependencies FEATS tags for each word.
        lemmas_tags: ``List[str]``, optional (default = None)
            The lemma tags for each word using edit scripts.
        dependencies ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.
        ids: ``List[str]``, optional (default = None)
            The universal dependencies ids for each word.
        multiword_ids: ``List[str]``, optional (default = None)
            The universal dependencies multiword ids (indicated by '-' in the id).
        multiword_forms: ``List[str]``, optional (default = None)
            The universal dependencies multiword forms (indicated by the multiword ids).

        Returns
        -------
        An instance containing words, ids, tags, dependency head tags and head
        indices as fields. The language identifier is stored in the metadata.
        """
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["words"] = tokens
        fields["lemmas"] = TextField([Token(l) for l in lemmas], self._token_indexers)

        tag_names = ["upos_tags", "xpos_tags", "pos_tags", "feats_tags", "lemmas_tags"]
        tag_fields = [upos_tags, xpos_tags, pos_tags, feats_tags, lemmas_tags]
        for name, field in zip(tag_names, tag_fields):
            if field:
                fields[name] = SequenceLabelField(field, tokens, label_namespace=name)

        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [tags for tags, _ in dependencies], tokens, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [indices for _, indices in dependencies], tokens, label_namespace="head_index_tags"
            )

        fields["metadata"] = MetadataField({
            "lang": lang,
            "words": words,
            "upos": upos_tags,
            "xpos": xpos_tags,
            "pos": pos_tags,
            "feats": feats_tags,
            "lemmas": lemmas,
            "lemmas_tags": lemmas_tags,
            "ids": ids,
            "multiword_ids": multiword_ids,
            "multiword_forms": multiword_forms
        })

        return Instance(fields)
