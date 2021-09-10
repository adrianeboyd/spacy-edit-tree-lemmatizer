from hypothesis import given
import hypothesis.strategies as st
import pickle
import srsly

from scripts.edittrees import EditTrees

def test_dutch():
    trees = EditTrees()
    tree = trees.add("deelt", "delen")
    assert trees.tree_str(tree) == "(i 0 3 () (i 0 2 (r '' 'l') (r 'lt' 'n')))"

    tree = trees.add("gedeeld", "delen")
    assert trees.tree_str(tree) == "(i 2 3 (r 'ge' '') (i 0 2 (r '' 'l') (r 'ld' 'n')))"

def test_pickle_roundtrip():
    trees = EditTrees()
    deelt_id = trees.add("deelt", "delen")
    gedeeld_id = trees.add("gedeeld", "delen")

    pickled = pickle.dumps(trees)
    unpickled = pickle.loads(pickled)

    # Verify that the nodes did not change.
    assert trees.size() == unpickled.size()
    for i in range(trees.size()):
        assert trees.tree_str(i) == unpickled.tree_str(i)


@given(st.text(), st.text())
def test_roundtrip(form, lemma):
    trees = EditTrees()
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma

@given(st.text(alphabet="ab"), st.text(alphabet="ab"))
def test_roundtrip_small_alphabet(form, lemma):
    # Test with small alphabets to have more overlap.
    trees = EditTrees()
    tree = trees.add(form, lemma)
    assert trees.apply(tree, form) == lemma
