from typing import Callable, Iterable, Optional, List, Tuple, Dict, Any
from itertools import islice
import numpy
import spacy
from spacy import Language, Vocab, Errors
from spacy.pipeline import TrainablePipe
from spacy.scorer import Scorer
from spacy.tokens.doc import Doc
from spacy.training import Example, validate_examples
import srsly
from thinc.loss import SequenceCategoricalCrossentropy
from thinc.model import Model

from .edittrees import EditTrees


@Language.factory(
    "edit_tree_lemmatizer",
    assigns=["token.lemma"],
    requires=[],
    default_config={"backoff": "form", "overwrite": False},
    default_score_weights={"lemma_acc": 1.0},
)
def make_edit_tree_lemmatizer(
    nlp: Language, name: str, model: Model, backoff: str, overwrite: bool = False
):
    """Construct an EditTreeLemmatizer component."""
    return EditTreeLemmatizer(nlp.vocab, model, name, backoff=backoff)


class EditTreeLemmatizer(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "lemmatizer",
        *,
        backoff: str = "form",
        overwrite: bool = False,
    ):
        self.vocab = vocab
        self.model = model
        self.name = name
        self.backoff = backoff
        self.overwrite = overwrite

        self.trees = EditTrees(vocab.strings)
        self.tree2label = dict()

        self.cfg = {"labels": []}

    def get_loss(self, examples, scores):
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

    def predict(self, docs, Doc=None):
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
            doc_guesses = doc_scores.argmax(axis=1)
            if not isinstance(doc_guesses, numpy.ndarray):
                doc_guesses = doc_guesses.get()

            for i, label in enumerate(doc_guesses):
                tree_id = self.labels[label]

                if self.trees.apply(tree_id, doc[i].text) is None:
                    doc_guesses[i] = -1
                else:
                    doc_guesses[i] = tree_id

            guesses.append(doc_guesses)

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
        labels: Optional[List[str]] = None,
    ):
        doc_sample = []
        label_sample = []

        # Construct the edit trees for all the examples.
        for example in get_examples():
            for token in example.reference:
                self._pair2label(token.text, token.lemma_)

        # Sample for the model.
        for example in islice(get_examples(), 10):
            doc_sample.append(example.x)
            gold_labels = []
            for token in example.reference:
                gold_label = self._pair2label(token.text, token.lemma_)
                gold_labels.append(
                    [1.0 if label == gold_label else 0.0 for label in self.labels]
                )

            label_sample.append(self.model.ops.asarray(gold_labels, dtype="float32"))

        self.model.initialize(X=doc_sample, Y=label_sample)

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

    def _pair2label(self, form, lemma):
        tree_id = self.trees.add(form, lemma)
        if tree_id not in self.tree2label:
            self.tree2label[tree_id] = len(self.labels)
            self.labels.append(tree_id)
        return self.tree2label[tree_id]
