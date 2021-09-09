from typing import Callable, Iterable, Optional, List, Tuple, Dict, Any

import numpy
from spacy import Language, Vocab, Errors
from spacy.pipeline import TrainablePipe
from spacy.scorer import Scorer
from spacy.tokens.doc import Doc
from spacy.training import Example, validate_examples
from thinc.loss import SequenceCategoricalCrossentropy
from thinc.model import Model

from .edittrees import EditTrees


@Language.factory(
    "edit_tree_lemmatizer",
    assigns=["token.lemma"],
    requires=[],
    default_score_weights={"lemma_acc": 1.0},
)
def make_edit_tree_lemmatizer(
        nlp: Language, name: str, model: Model
):
    """Construct a RelationExtractor component."""
    return EditTreeLemmatizer(nlp.vocab, model, name)


class EditTreeLemmatizer(TrainablePipe):
    def __init__(self, vocab: Vocab, model: Model, name: str = "lemmatizer"):
        self.vocab = vocab
        self.model = model
        self.name = name
        self.trees = EditTrees()
        self.tree2label = dict()

        self.cfg = {"labels": []}

    def get_loss(self, examples, scores):
        validate_examples(examples, "EditTreeLemmatizer.get_loss")
        loss_func = SequenceCategoricalCrossentropy(names=self.labels, normalize=False)

        truths = []
        for eg in examples:
            ex_truths = []
            # eg_truths = [tag if tag is not "" else None for tag in eg.get_aligned("TAG", as_string=True)]
            for (predicted, gold_lemma) in zip(eg.predicted, eg.get_aligned("LEMMA", as_string=True)):
                if gold_lemma is None:
                    gold_lemma = predicted.text # ???

                tree_id = self.trees.add(predicted.text, gold_lemma)
                # XXX: 0 is not actually the null tree
                label = self.tree2label.get(tree_id, 0)
                ex_truths.append(label)
            truths.append(ex_truths)

        #truths = self.model.ops.asarray(truths)
        #print(self.trees.size())

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

        guesses = self._scores2guesses(scores)
        assert len(guesses) == len(docs)

        return guesses

    def set_annotations(self, docs, batch_lemma_ids):
        if isinstance(docs, Doc):
            docs = [docs]

        for i, doc in enumerate(docs):
            doc_lemma_ids = batch_lemma_ids[i]
            if hasattr(doc_lemma_ids, "get"):
                doc_lemma_ids = doc_lemma_ids.get()
            for j, tree_id in enumerate(doc_lemma_ids):
                if doc[j].lemma_ == '':
                    node_id = self.labels[tree_id]
                    doc[j].lemma_ = self.trees.apply(node_id, doc[j].text)

    def _scores2guesses(self, scores):
        guesses = []
        for doc_scores in scores:
            doc_guesses = doc_scores.argmax(axis=1)
            if not isinstance(doc_guesses, numpy.ndarray):
                doc_guesses = doc_guesses.get()
            guesses.append(doc_guesses)
        return guesses


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


    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Language = None, labels: Optional[List[str]] = None):
        doc_sample = []
        label_sample = []

        for example in get_examples():
            doc_sample.append(example.x)
            for token in example.y:
                labels = []
                tree_id = self.trees.add(token.text, token.lemma_)
                if not tree_id in self.tree2label:
                    self.tree2label[tree_id] = len(self.labels)
                    self.labels.append(tree_id)
                label = self.tree2label[tree_id]
                labels.append(label)
                label_sample.append(self.model.ops.asarray(labels, dtype="float32"))

        self.model.initialize(X=doc_sample, Y=label_sample)
