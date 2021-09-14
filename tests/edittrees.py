from hypothesis import given
import hypothesis.strategies as st
import pytest
import pickle

from scripts.edittrees import EditTrees
from spacy.strings import StringStore
from spacy.util import make_tempdir


def test_dutch():
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add("deelt", "delen")
    assert trees.tree_str(tree) == "(i 0 3 () (i 0 2 (r '' 'l') (r 'lt' 'n')))"

    tree = trees.add("gedeeld", "delen")
    assert trees.tree_str(tree) == "(i 2 3 (r 'ge' '') (i 0 2 (r '' 'l') (r 'ld' 'n')))"


@pytest.mark.skip(msg="Removing pickling?")
def test_pickle_roundtrip():
    strings = StringStore()
    trees = EditTrees(strings)
    deelt_id = trees.add("deelt", "delen")
    gedeeld_id = trees.add("gedeeld", "delen")

    pickled = pickle.dumps(trees)
    unpickled = pickle.loads(pickled)

    # Verify that the nodes did not change.
    assert trees.size() == unpickled.size()
    for i in range(trees.size()):
        assert trees.tree_str(i) == unpickled.tree_str(i)


def test_from_to_bytes():
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add("deelt", "delen")
    trees.add("gedeeld", "delen")

    b = trees.to_bytes()

    trees2 = EditTrees(strings)
    trees2.from_bytes(b)

    # Verify that the nodes did not change.
    assert trees.size() == trees2.size()
    for i in range(trees.size()):
        assert trees.tree_str(i) == trees2.tree_str(i)


def test_from_to_disk():
    strings = StringStore()
    trees = EditTrees(strings)
    trees.add("deelt", "delen")
    trees.add("gedeeld", "delen")

    trees2 = EditTrees(strings)
    with make_tempdir() as temp_dir:
        trees_file = temp_dir / "edittrees.bin"
        trees.to_disk(trees_file)
        trees2 = trees2.from_disk(trees_file)

    # Verify that the nodes did not change.
    assert trees.size() == trees2.size()
    for i in range(trees.size()):
        assert trees.tree_str(i) == trees2.tree_str(i)


@given(st.text(), st.text())
def test_roundtrip(form, lemma):
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma


@given(st.text(alphabet="ab"), st.text(alphabet="ab"))
def test_roundtrip_small_alphabet(form, lemma):
    # Test with small alphabets to have more overlap.
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma
