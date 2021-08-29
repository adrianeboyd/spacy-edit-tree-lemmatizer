# cython: infer_types=True, profile=True, binding=True

from preshed.maps import PreshMap
from spacy.strings import StringStore
from cymem.cymem cimport Pool
from libc.stdint cimport uint32_t, uint64_t
from spacy.typedefs cimport attr_t, hash_t, len_t
from preshed.maps cimport PreshMap
from libc.string cimport memset
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from spacy.strings cimport StringStore
from libcpp.vector cimport vector
import numpy as np
from edittrees cimport lcs_is_empty

from edittrees cimport EditTrees, EditTreeNodeC

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



cdef class EditTrees:
    def __init__(self):
        self.nodes = vector[EditTreeNodeC]()
        self.map = PreshMap()
        self.strings = StringStore()

    def add(self, form: str, lemma: str) -> int:
        return self.build(form, lemma)


    cdef uint64_t build(self, str form, str lemma):
        cdef EditTreeNodeC node
        cdef uint64_t node_id, left, right

        lcs = find_lcs(form, lemma)

        if lcs_is_empty(lcs):
            node = edit_tree_node_new_replacement(self.strings.add(form), self.strings.add(lemma))
        else:
            left = 0
            if lcs.source_begin != 0 or lcs.target_begin != 0:
                left = self.build(form[:lcs.source_begin], lemma[:lcs.target_begin])

            right = 0
            if lcs.source_end != len(form) or lcs.target_end != len(lemma):
                right = self.build(form[lcs.source_end:], lemma[lcs.target_end:])

            node = edit_tree_node_new_interior(lcs.source_begin, len(form) - lcs.source_end, left, right)

        cdef hash_t hash = edit_tree_node_hash(node)
        node_id = <uint64_t>self.map.get(hash)
        if node_id == 0:
            self.nodes.push_back(node)
            node_id = self.nodes.size()
            self.map.set(hash, <void *>node_id)

        return node_id

    cpdef str apply(self, uint64_t tree, str form):
        lemma_pieces = []
        self._apply(tree, form, lemma_pieces)
        return "".join(lemma_pieces)

    cdef _apply(self, uint64_t tree, str form_part, list lemma_pieces):
        cdef EditTreeNodeC node = self.nodes[tree - 1]
        cdef InteriorNodeC interior
        cdef int suffix_start

        if node.is_interior_node:
            interior = node.inner.interior_node
            suffix_start = len(form_part) - interior.suffix_len

            if interior.left != 0:
                self._apply(interior.left, form_part[:interior.prefix_len], lemma_pieces)

            lemma_pieces.append(form_part[interior.prefix_len:suffix_start])

            if interior.right != 0:
                self._apply(interior.right, form_part[suffix_start:], lemma_pieces)
        else:
            lemma_pieces.append(self.strings[node.inner.replacement_node.replacement])

    cpdef tree_str(self, uint32_t tree):
        assert tree > 0

        cdef EditTreeNodeC node = self.nodes[tree - 1]
        cdef InteriorNodeC interior
        cdef ReplacementNodeC replacement

        if node.is_interior_node:
            interior = node.inner.interior_node

            left = "()"
            if interior.left != 0:
                left = self.tree_str(interior.left)

            right = "()"
            if interior.right != 0:
                right = self.tree_str(interior.right)

            return f"(i {interior.prefix_len} {interior.suffix_len} {left} {right})"

        replacement = node.inner.replacement_node

        return f"(r '{self.strings[replacement.replacee]}' '{self.strings[replacement.replacement]}')"

    def __reduce__(self):
        return (unpickle_edit_trees, (self.nodes, self.strings), None, None)

def unpickle_edit_trees(nodes, strings):
    cdef EditTreeNodeC c_node
    cdef uint64_t node_id
    cdef hash_t node_hash

    trees = EditTrees()

    trees.strings = strings

    for node in nodes:
        if node['is_interior_node']:
            del node['inner']['replacement_node']
        else:
            del node['inner']['interior_node']
        c_node = node
        trees.nodes.push_back(c_node)
        node_id = trees.nodes.size()
        node_hash = edit_tree_node_hash(c_node)
        trees.map.set(node_hash, <void *>node_id)

    return trees
