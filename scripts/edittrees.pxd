from libc.stdint cimport uint32_t, uint64_t
from spacy.typedefs cimport attr_t, hash_t, len_t
from spacy.strings cimport StringStore
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef extern from "<algorithm>" namespace "std" nogil:
    void swap[T](T& a, T& b) except + # Only available in Cython 3.

cdef uint32_t NULL_TREE_ID

cdef struct InteriorNodeC:
    len_t prefix_len
    len_t suffix_len
    uint32_t prefix_tree
    uint32_t suffix_tree

cdef struct ReplacementNodeC:
    attr_t replacee
    attr_t replacement

cdef union NodeC:
    InteriorNodeC interior_node
    ReplacementNodeC replacement_node

cdef struct EditTreeC:
    bint is_interior_node
    NodeC inner

cdef inline EditTreeC edittree_new_interior(len_t prefix_len, len_t suffix_len, uint32_t prefix_tree, uint32_t suffix_tree):
    cdef InteriorNodeC interior_node = InteriorNodeC(prefix_len=prefix_len, suffix_len=suffix_len, prefix_tree=prefix_tree, suffix_tree=suffix_tree)
    cdef NodeC inner = NodeC(interior_node=interior_node)
    cdef EditTreeC node = EditTreeC(is_interior_node=True, inner=inner)
    return node

cdef inline EditTreeC edittree_new_replacement(attr_t replacee, attr_t replacement):
    cdef EditTreeC node
    node.is_interior_node = False
    node.inner.replacement_node.replacee = replacee
    node.inner.replacement_node.replacement = replacement
    return node

cdef inline uint64_t edittree_hash(EditTreeC tree):
    cdef InteriorNodeC interior
    cdef ReplacementNodeC replacement

    if tree.is_interior_node:
        interior = tree.inner.interior_node
        return hash((interior.prefix_len, interior.suffix_len, interior.prefix_tree, interior.suffix_tree))
    else:
        replacement = tree.inner.replacement_node
        return hash((replacement.replacee, replacement.replacement))

cdef struct LCS:
    int source_begin
    int source_end
    int target_begin
    int target_end

cdef inline bint lcs_is_empty(LCS lcs):
    return lcs.source_begin == 0 and lcs.source_end == 0 and lcs.target_begin == 0 and lcs.target_end == 0

cdef class EditTrees:
    cdef vector[EditTreeC] trees
    cdef unordered_map[hash_t, uint32_t] map
    cdef StringStore strings

    cpdef str apply(self, uint32_t tree_id, str form)
    cdef _apply(self, uint32_t tree_id, str form_part, list lemma_pieces)
    cdef uint32_t build(self, str form, str lemma)
    cpdef tree_str(self, uint32_t tree_id)
