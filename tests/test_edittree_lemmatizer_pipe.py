import pytest
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example

from scripts.edittree_lemmatizer_pipe import EditTreeLemmatizer


TRAIN_DATA = [
    ("She likes green eggs", {"lemmas": ["she", "like", "green", "egg"]}),
    ("Eat blue ham", {"lemmas": ["eat", "blue", "ham"]}),
]

PARTIAL_DATA = [
    # partial annotation
    ("She likes green eggs", {"lemmas": ["", "like", "green", ""]}),
    # misaligned partial annotation
    (
        "He hates green eggs",
        {
            "words": ["He", "hat", "es", "green", "eggs"],
            "lemmas": ["", "hat", "e", "green", ""],
        },
    ),
]


def test_initialize_examples():
    nlp = Language()
    lemmatizer = nlp.add_pipe("edit_tree_lemmatizer")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    # you shouldn't really call this more than once, but for testing it should be fine
    nlp.initialize(get_examples=lambda: train_examples)
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda: None)
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda: train_examples[0])
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=lambda: [])
    with pytest.raises(TypeError):
        nlp.initialize(get_examples=train_examples)


def test_no_data():
    # Test that the lemmatizer provides a nice error when there's no tagging data / labels
    TEXTCAT_DATA = [
        ("I'm so happy.", {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}),
        ("I'm so angry", {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}),
    ]
    nlp = English()
    nlp.add_pipe("edit_tree_lemmatizer")
    nlp.add_pipe("textcat")

    train_examples = []
    for t in TEXTCAT_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    with pytest.raises(ValueError):
        nlp.initialize(get_examples=lambda: train_examples)


def test_incomplete_data():
    # Test that the lemmatizer works with incomplete information
    nlp = English()
    nlp.add_pipe("edit_tree_lemmatizer")
    train_examples = []
    for t in PARTIAL_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses["edit_tree_lemmatizer"] < 0.00001

    # test the trained model
    test_text = "She likes blue eggs"
    doc = nlp(test_text)
    assert doc[1].lemma_ == "like"
    assert doc[2].lemma_ == "blue"


def test_overfitting_IO():
    nlp = English()
    lemmatizer = nlp.add_pipe("edit_tree_lemmatizer")
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))

    optimizer = nlp.initialize(get_examples=lambda: train_examples)

    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses["edit_tree_lemmatizer"] < 0.00001

    test_text = "She likes blue eggs"
    doc = nlp(test_text)
    assert doc[0].lemma_ == "she"
    assert doc[1].lemma_ == "like"
    assert doc[2].lemma_ == "blue"
    assert doc[3].lemma_ == "egg"

    # Test that the model still makes correct predictions after an IO roundtrip
    with util.make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        assert doc2[0].lemma_ == "she"
        assert doc2[1].lemma_ == "like"
        assert doc2[2].lemma_ == "blue"
        assert doc2[3].lemma_ == "egg"


def test_lemmatizer_requires_labels():
    nlp = English()
    nlp.add_pipe("edit_tree_lemmatizer")
    with pytest.raises(ValueError):
        nlp.initialize()
