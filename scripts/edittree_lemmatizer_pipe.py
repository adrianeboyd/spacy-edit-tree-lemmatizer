from typing import Callable, Iterable, Optional, List, Tuple
from spacy import Language, Vocab
from spacy.pipeline import TrainablePipe
from spacy.training import Example
from thinc.model import Model

from .edittrees import EditTrees


@Language.factory(
    "edit_tree_lemmatizer",
    assigns=["doc._.lemma"],
    requires=[],
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

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return self.cfg["labels"]

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
