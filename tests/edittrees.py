from hypothesis import given
import hypothesis.strategies as st

from scripts.edittrees import EditTrees

def test_dutch():
    trees = EditTrees()
    tree = trees.add("deelt", "delen")
    assert trees.tree_str(tree) == "(i 0 3 () (i 0 2 (r '' 'l' (r 'lt' 'n'))"

    tree = trees.add("gedeeld", "delen")
    assert trees.tree_str(tree) == "(i 2 3 (r 'ge' '' (i 0 2 (r '' 'l' (r 'ld' 'n'))"


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
