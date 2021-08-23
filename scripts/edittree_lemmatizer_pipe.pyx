# cython: infer_types=True, profile=True, binding=True
import numpy as np
from typing import Callable, Iterable, Optional, List
from spacy import Language, Vocab
from spacy.pipeline import TrainablePipe
from spacy.training import Example
from thinc.model import Model


@Language.factory(
    "edit_tree_lemmatizer",
    requires=[],
    assigns=["doc._.lemma"],
)
def make_edit_tree_lemmatizer(
        nlp: Language, name: str
):
    """Construct a RelationExtractor component."""
    return EditTreeLemmatizer(nlp.vocab, name)

class EditTreeLemmatizer(TrainablePipe):
    def __init__(self, vocab: Vocab, model: Model, name: str = "lemma"):
        self.vocab = vocab
        self.model = model
        self.name = name

    def initialize(self, get_examples: Callable[[], Iterable[Example]], *, nlp: Language = None, labels: Optional[List[str]] = None):
        trees = set()

        for example in get_examples():
            for token in example.y:
                trees.add(build_edit_tree(token.text, token.lemma_))
                #lcs = find_lcs(token.orth_, token.lemma_)
                #print("LCS: %s %s" % (token.text[lcs.source_begin:lcs.source_end], token.lemma_[lcs.source_begin:lcs.source_end]))

        print(trees)
        print(len(trees))

cdef class TreeNode:
    pass

cdef class EditTree(TreeNode):
    cdef TreeNode left
    cdef TreeNode right
    cdef int prefix_len
    cdef int suffix_len

    def __init__(self, prefix_len: int, suffix_len: int, left: TreeNode, right: TreeNode):
        self.left = left
        self.right = right
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len

    cpdef apply(self, form_part: str, lemma_pieces: [str]):
        cdef int suffix_start

        # Append prefix edits.
        if self.left:
            self.left.apply(form_part[:self.prefix_len], lemma_pieces)

        suffix_start = len(form_part) - self.suffix_len

        # Append common subsequence.
        lemma_pieces.append(form_part[self.prefix_len:suffix_start])

        # Append suffix edits.
        if self.right:
            self.right.apply(form_part[suffix_start:], lemma_pieces)

    @property
    def left(self):
        return self.left

    @property
    def prefix_len(self):
        return self.prefix_len

    @property
    def right(self):
        return self.right

    @property
    def suffix_len(self):
        return self.suffix_len

    def __str__(self):
        return f"(e {self.prefix_len} {self.suffix_len} {self.left} {self.right})"

    def __eq__(self, other):
        return type(other) is EditTree and self.prefix_len == other.prefix_len and \
            self.suffix_len == other.suffix_len and self.left == other.left and \
            self.right == other.right

    def __hash__(self) -> int:
        return hash((self.prefix_len, self.suffix_len, self.left, self.right))


cdef class ReplacementNode(TreeNode):
    cdef public str replacee
    cdef public str replacement

    def __init__(self, replacee: str, replacement: str):
        self.replacee = replacee
        self.replacement = replacement

    cpdef apply(self, lemma_part: str, lemma_pieces: [str]):
        lemma_pieces.append(self.replacement)

    def __eq__(self, other):
        return type(other) is ReplacementNode and self.replacee == other.replacee and \
               self.replacement == other.replacement

    def __str__(self):
        return f"(r '{self.replacee}' '{self.replacement}')"

    def __hash__(self) -> int:
        return hash((self.replacee, self.replacement))

cdef class LCS:
    cdef int source_begin
    cdef int source_end
    cdef int target_begin
    cdef int target_end

    def __init__(self):
        self.source_begin = 0
        self.source_end = 0
        self.target_begin = 0
        self.target_end = 0

    def empty(self) -> bool:
        return self.source_begin == self.source_end

cpdef build_edit_tree(form: str, lemma: str):
    """
    Build an edit tree for rewriting a form into a lemma.
    
    :param form: 
    :param lemma: 
    :return: 
    """
    lcs = find_lcs(form, lemma)

    if lcs.empty():
        return ReplacementNode(form, lemma)

    left = None
    if lcs.source_begin != 0 or lcs.target_begin != 0:
        left = build_edit_tree(form[:lcs.source_begin], lemma[:lcs.target_begin])

    right = None
    if lcs.source_end != len(form) or lcs.target_end != len(lemma):
        right = build_edit_tree(form[lcs.source_end:], lemma[lcs.target_end:])

    return EditTree(lcs.source_begin, len(form) - lcs.source_end, left, right)

cdef LCS find_lcs(source: str, target: str):
    """
    Find the longest common subsequence (LCS) between two strings. If there are
    multiple LCSes, only one of them is returned.
    
    :param source: The first string.
    :param target: The second string.
    :return: The spans of the longest common subsequences.
    """
    cdef Py_ssize_t source_len = len(source)
    cdef Py_ssize_t target_len = len(target)
    cdef int longest_align = 0;
    cdef int lcs_source_start = 0, lcs_source_end = 0

    lcs = LCS()

    cdef int[:, :] align_lens = np.zeros((source_len, target_len), dtype=np.intc)

    for source_idx in range(source_len):
        for target_idx in range(target_len):
            if source[source_idx] == target[target_idx]:
                if source_idx == 0 or target_idx == 0:
                    align_lens[source_idx, target_idx] = 1
                else:
                    align_lens[source_idx, target_idx] = align_lens[source_idx - 1, target_idx - 1] + 1

                if align_lens[source_idx, target_idx] > longest_align:
                    longest_align = align_lens[source_idx, target_idx]
                    lcs.source_begin = source_idx - longest_align + 1
                    lcs.source_end =  source_idx + 1
                    lcs.target_begin = target_idx - longest_align + 1
                    lcs.target_end =  target_idx + 1
            else:
                # No match, we start with a zero-length LCS.
                align_lens[source_idx, target_idx] = 0

    return lcs