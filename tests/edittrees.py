from hypothesis import given
import hypothesis.strategies as st
from spacy.strings import StringStore
from spacy.util import make_tempdir

from scripts.edittrees import EditTrees


def test_dutch():
    strings = StringStore()
    trees = EditTrees(strings)
    tree = trees.add("deelt", "delen")
    assert trees.tree_str(tree) == "(i 0 3 () (i 0 2 (r '' 'l') (r 'lt' 'n')))"

    tree = trees.add("gedeeld", "delen")
    assert trees.tree_str(tree) == "(i 2 3 (r 'ge' '') (i 0 2 (r '' 'l') (r 'ld' 'n')))"


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

    # Reinserting the same trees should not add new nodes.
    trees2.add("deelt", "delen")
    trees2.add("gedeeld", "delen")
    assert trees.size() == trees2.size()


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

    # Reinserting the same trees should not add new nodes.
    trees2.add("deelt", "delen")
    trees2.add("gedeeld", "delen")
    assert trees.size() == trees2.size()


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
