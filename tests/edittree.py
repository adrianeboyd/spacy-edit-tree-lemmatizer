from cymem.cymem import Pool
from hypothesis import given
import hypothesis.strategies as st
from spacy.vocab import Vocab

from scripts.edittree import EditTree


@given(st.text(), st.text())
def test_roundtrip(form, lemma):
    pool = Pool()
    vocab = Vocab()
    tree = EditTree(pool, vocab, form, lemma)
    print(f"form: {form}, lemma: {lemma}, tree: {tree}")
    assert tree.apply(form) == lemma


@given(st.text(alphabet="ab"), st.text(alphabet="ab"))
def test_roundtrip_small_alphabet(form, lemma):
    # Test with small alphabets to have more overlap.
    pool = Pool()
    vocab = Vocab()
    tree = EditTree(pool, vocab, form, lemma)
    print(f"form: '{form}', lemma: '{lemma}', tree: {tree}")
    assert tree.apply(form) == lemma