# cython: infer_types=True, profile=True, binding=True

from libc.string cimport memset
import numpy as np
from spacy.vocab import Vocab

from edittree cimport EditTreeNodeC, edit_tree_node_new_interior, edit_tree_node_new_replacement

cdef struct LCS:
    int source_begin
    int source_end
    int target_begin
    int target_end

cdef lcs_is_empty(lcs: LCS):
    return lcs.source_begin == 0 and lcs.source_end == 0 and lcs.target_begin == 0 and lcs.target_end == 0

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
    cdef LCS lcs

    memset(&lcs, 0, sizeof(lcs))

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

cdef class EditTree:
    cdef Vocab vocab;
    cdef EditTreeNodeC *root;

    def __init__(self, vocab: Vocab, form: str, lemma: str):
        self.vocab = vocab
        self.root = EditTree.build(vocab, form, lemma)

    cpdef str apply(self, form: str):
        lemma_pieces = []
        self.apply_(self.root, form, lemma_pieces)
        return "".join(lemma_pieces)

    cdef apply_(self, EditTreeNodeC *node, str form_part, list lemma_pieces):
        assert node != NULL

        if not node.is_interior_node:
            lemma_pieces.append(self.vocab.strings[node.node.replacement_node.replacement])
            return

        cdef prefix_len = node.node.interior_node.prefix_len
        cdef suffix_len = node.node.interior_node.suffix_len
        cdef suffix_start = len(form_part) - suffix_len

        # Append prefix edits.
        if node.node.interior_node.left != NULL:
            self.apply_(node.node.interior_node.left, form_part[:prefix_len], lemma_pieces)

        # Append common subsequence.
        lemma_pieces.append(form_part[prefix_len:suffix_start])

        # Append suffix edits.
        if node.node.interior_node.right != NULL:
            self.apply_(node.node.interior_node.right, form_part[suffix_start:], lemma_pieces)

    @staticmethod
    cdef EditTreeNodeC *build(vocab: Vocab, form: str, lemma: str):
        cdef EditTreeNodeC *node

        lcs = find_lcs(form, lemma)

        if lcs_is_empty(lcs):
            return edit_tree_node_new_replacement(vocab.strings.add(form), vocab.strings.add(lemma))

        cdef EditTreeNodeC *left = NULL
        if lcs.source_begin != 0 or lcs.target_begin != 0:
            left = EditTree.build(vocab, form[:lcs.source_begin], lemma[:lcs.target_begin])

        cdef EditTreeNodeC *right = NULL
        if lcs.source_end != len(form) or lcs.target_end != len(lemma):
            right = EditTree.build(vocab, form[lcs.source_end:], lemma[lcs.target_end:])

        return edit_tree_node_new_interior(lcs.source_begin, len(form) - lcs.source_end, left, right)

    def __str__(self):
        return self._str(self.root)

    cdef _str(self, EditTreeNodeC *node):
        if node.is_interior_node:
            prefix_len = node.node.interior_node.prefix_len
            suffix_len = node.node.interior_node.suffix_len

            left = "None"
            if node.node.interior_node.left != NULL:
                left = self._str(node.node.interior_node.left)

            right = "None"
            if node.node.interior_node.right != NULL:
                left = self._str(node.node.interior_node.right)

            return f"(i {prefix_len} {suffix_len} {left} {right})"
        else:
            replacee = self.vocab.strings[node.node.replacement_node.replacee]
            replacement = self.vocab.strings[node.node.replacement_node.replacement]
            return f"(r '{replacee}' '{replacement}')"