from typing import Callable, Iterable, Optional, List, Tuple, Dict, Any
from itertools import islice
import numpy as np
import spacy
from spacy import Language, Vocab, Errors
from spacy.pipeline import TrainablePipe
from spacy.scorer import Scorer
from spacy.tokens.doc import Doc
from spacy.training import Example, validate_examples, validate_get_examples
import srsly
from thinc.api import Config, Model, SequenceCategoricalCrossentropy
from thinc.types import Floats2d, Ints2d

from .edit_trees import EditTrees

default_model_config = """
[model]
@architectures = "spacy.Tagger.v1"

[model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
pretrained_vectors = null
width = 96
depth = 4
embed_size = 2000
window_size = 1
maxout_pieces = 3
subword_features = true
"""
DEFAULT_EDIT_TREE_LEMMATIZER_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory(
    "edit_tree_lemmatizer",
    assigns=["token.lemma"],
    requires=[],
    default_config={
        "model": DEFAULT_EDIT_TREE_LEMMATIZER_MODEL,
        "backoff": "form",
        "overwrite": False,
        "top_k": 1,
    },
    default_score_weights={"lemma_acc": 1.0},
)
def make_edit_tree_lemmatizer(
    nlp: Language,
    name: str,
    model: Model,
    backoff: str,
    overwrite: bool = False,
    top_k: int = 1,
):
    """Construct an EditTreeLemmatizer component."""
    return EditTreeLemmatizer(
        nlp.vocab, model, name, backoff=backoff, overwrite=overwrite, top_k=top_k
    )


class EditTreeLemmatizer(TrainablePipe):
    """
    Lemmatizer that lemmatizes each word using a predicted edit tree.
    """

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "lemmatizer",
        *,
        backoff: str = "form",
        overwrite: bool = False,
        top_k: int = 1,
    ):
        """
        Construct and edit tree lemmatizer.

        backoff (str): back-off to use when the predicted edit trees
            are not applicable. Must be one of "form" (use the form
            as the lemma) or None (leave the lemma unset).
        overwrite (bool): overwrite existing lemma annotations.
        top_k (int): try to apply at most the k most probable edit trees.
        """
        self.vocab = vocab
        self.model = model
        self.name = name
        self.backoff = backoff
        self.overwrite = overwrite
        self.top_k = top_k

        self.trees = EditTrees(vocab.strings)
        self.tree2label = dict()

        self.cfg = {"labels": []}

    def get_loss(
        self, examples: Iterable[Example], scores
    ) -> Tuple[float, List[Floats2d]]:
        validate_examples(examples, "EditTreeLemmatizer.get_loss")
        loss_func = SequenceCategoricalCrossentropy(normalize=False, missing_value=-1)

        truths = []
        for eg in examples:
            eg_truths = []
            for (predicted, gold_lemma) in zip(
                eg.predicted, eg.get_aligned("LEMMA", as_string=True)
            ):
                if gold_lemma is None:
                    label = -1
                else:
                    tree_id = self.trees.add(predicted.text, gold_lemma)
                    label = self.tree2label.get(tree_id, 0)
                eg_truths.append(label)

            truths.append(eg_truths)

        d_scores, loss = loss_func(scores, truths)
        if self.model.ops.xp.isnan(loss):
            raise ValueError(Errors.E910.format(name=self.name))

        return float(loss), d_scores

    def predict(self, docs: Iterable[Doc]) -> List[Ints2d]:
        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            n_labels = len(self.labels)
            guesses = [self.model.ops.alloc((0, n_labels)) for doc in docs]
            assert len(guesses) == len(docs)
            return guesses

        scores = self.model.predict(docs)
        assert len(scores) == len(docs)

        guesses = self._scores2guesses(docs, scores)
        assert len(guesses) == len(docs)

        return guesses

    def _scores2guesses(self, docs, scores):
        guesses = []
        for doc, doc_scores in zip(docs, scores):
            if self.top_k == 1:
                doc_guesses = doc_scores.argmax(axis=1).reshape(-1, 1)
            else:
                doc_guesses = np.argsort(doc_scores)[..., : -self.top_k - 1 : -1]

            if not isinstance(doc_guesses, np.ndarray):
                doc_guesses = doc_guesses.get()

            doc_compat_guesses = []
            for token, candidates in zip(doc, doc_guesses):
                tree_id = -1
                for candidate in candidates:
                    candidate_tree_id = self.labels[candidate]

                    if self.trees.apply(candidate_tree_id, token.text) is not None:
                        tree_id = candidate_tree_id
                        break
                doc_compat_guesses.append(tree_id)

            guesses.append(np.array(doc_compat_guesses))

        return guesses

    def set_annotations(self, docs, batch_tree_ids):
        if isinstance(docs, Doc):
            docs = [docs]

        for i, doc in enumerate(docs):
            doc_tree_ids = batch_tree_ids[i]
            if hasattr(doc_tree_ids, "get"):
                doc_tree_ids = doc_tree_ids.get()
            for j, tree_id in enumerate(doc_tree_ids):
                if self.overwrite or doc[j].lemma == 0:
                    # If no applicable tree could be found during prediction,
                    # the special identifier -1 is used. Otherwise the tree
                    # is guaranteed to be applicable.
                    if tree_id == -1:
                        if self.backoff == "form":
                            doc[j].lemma_ = doc[j].text
                    else:
                        lemma = self.trees.apply(tree_id, doc[j].text)
                        doc[j].lemma_ = lemma

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return self.cfg["labels"]

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples.

        examples (Iterable[Example]): The examples to score.
        RETURNS (Dict[str, Any]): The scores.

        DOCS: https://spacy.io/api/lemmatizer#score
        """
        validate_examples(examples, "EditTreeLemmatizer.score")
        return Scorer.score_token_attr(examples, "lemma", **kwargs)

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
    ):
        validate_get_examples(get_examples, "EditTreeLemmatizer.initialize")

        # Construct the edit trees for all the examples.
        for example in get_examples():
            for token in example.reference:
                if token.lemma != 0:
                    self._pair2label(token.text, token.lemma_, add_label=True)

        # Sample for the model.
        doc_sample = []
        label_sample = []
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
            gold_labels = []
            for token in example.reference:
                if token.lemma == 0:
                    gold_label = None
                else:
                    gold_label = self._pair2label(token.text, token.lemma_)

                gold_labels.append(
                    [1.0 if label == gold_label else 0.0 for label in self.labels]
                )

            label_sample.append(self.model.ops.asarray(gold_labels, dtype="float32"))

        self._require_labels()
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        assert len(label_sample) > 0, Errors.E923.format(name=self.name)

        self.model.initialize(X=doc_sample, Y=label_sample)

    def from_bytes(self, bytes_data, *, exclude=tuple()):
        deserializers = {
            "cfg": lambda b: self.cfg.update(srsly.json_loads(b)),
            "model": lambda b: self.model.from_bytes(b),
            "vocab": lambda b: self.vocab.from_bytes(b, exclude=exclude),
            "trees": lambda b: self.trees.from_bytes(b),
        }

        spacy.util.from_bytes(bytes_data, deserializers, exclude)

        return self

    def to_bytes(self, *, exclude=tuple()):
        serializers = {
            "cfg": lambda: srsly.json_dumps(self.cfg),
            "model": lambda: self.model.to_bytes(),
            "vocab": lambda: self.vocab.to_bytes(exclude=exclude),
            "trees": lambda: self.trees.to_bytes(),
        }

        return spacy.util.to_bytes(serializers, exclude)

    def to_disk(self, path, exclude=tuple()):
        path = spacy.util.ensure_path(path)
        serializers = {
            "cfg": lambda p: srsly.write_json(p, self.cfg),
            "model": lambda p: self.model.to_disk(p),
            "vocab": lambda p: self.vocab.to_disk(p, exclude=exclude),
            "trees": lambda p: self.trees.to_disk(p),
        }
        spacy.util.to_disk(path, serializers, exclude)

    def from_disk(self, path, exclude=tuple()):
        def load_model(p):
            try:
                with open(p, "rb") as mfile:
                    self.model.from_bytes(mfile.read())
            except AttributeError:
                raise ValueError(Errors.E149) from None

        deserializers = {
            "cfg": lambda p: self.cfg.update(srsly.read_json(p)),
            "model": load_model,
            "vocab": lambda p: self.vocab.from_disk(p, exclude=exclude),
            "trees": lambda p: self.trees.from_disk(p),
        }

        spacy.util.from_disk(path, deserializers, exclude)
        return self

    def _pair2label(self, form, lemma, add_label=False):
        """
        Look up the edit tree identifier for a form/label pair. If the edit
        tree is unknown and "add_label" is set, the edit tree will be added to
        the labels.
        """
        tree_id = self.trees.add(form, lemma)
        if tree_id not in self.tree2label:
            if not add_label:
                return None

            self.tree2label[tree_id] = len(self.labels)
            self.labels.append(tree_id)
        return self.tree2label[tree_id]
