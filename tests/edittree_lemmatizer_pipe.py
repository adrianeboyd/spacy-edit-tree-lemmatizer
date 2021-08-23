from hypothesis import given
import hypothesis.strategies as st
from scripts.edittree_lemmatizer_pipe import build_edit_tree, EditTree, ReplacementNode


def test_dutch():
    tree = build_edit_tree("deelt", "delen")
    print(tree)
    assert build_edit_tree("deelt", "delen") == \
           EditTree(0, 3, None, EditTree(0, 2, ReplacementNode('', 'l'), ReplacementNode('lt', 'n')))
    assert build_edit_tree("gedeeld", "delen") == \
           EditTree(2, 3, ReplacementNode('ge', ''), EditTree(0, 2, ReplacementNode('', 'l'), ReplacementNode('ld', 'n')))


def test_apply():
    pieces = []
    EditTree(0, 3, None, EditTree(0, 2, ReplacementNode('', 'l'), ReplacementNode('lt', 'n'))).apply("deelt", pieces)
    assert ''.join(pieces) == "delen"


@given(st.text(), st.text())
def test_roundtrip(form, lemma):
    tree = build_edit_tree(form, lemma)
    lemma_pieces = []
    tree.apply(form, lemma_pieces)
    assert ''.join(lemma_pieces) == lemma


@given(st.text(alphabet="ab"), st.text(alphabet="ab"))
def test_roundtrip_small_alphabet(form, lemma):
    # Test with small alphabets to have more overlap.
    tree = build_edit_tree(form, lemma)
    lemma_pieces = []
    tree.apply(form, lemma_pieces)
    assert ''.join(lemma_pieces) == lemma
