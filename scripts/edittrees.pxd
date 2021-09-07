from libc.stdint cimport uint32_t, uint64_t
from spacy.typedefs cimport attr_t, hash_t, len_t
from spacy.strings cimport StringStore
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef extern from "<algorithm>" namespace "std" nogil:
    void swap[T](T& a, T& b) except + # Only available in Cython 3.

cdef uint32_t NULL_NODE_ID

cdef struct InteriorNodeC:
    len_t prefix_len
    len_t suffix_len
    uint32_t left
    uint32_t right

cdef struct ReplacementNodeC:
    attr_t replacee
    attr_t replacement

cdef union NodeC:
    InteriorNodeC interior_node
    ReplacementNodeC replacement_node

cdef struct EditTreeNodeC:
    bint is_interior_node
    NodeC inner

cdef inline EditTreeNodeC edit_tree_node_new_interior(len_t prefix_len, len_t suffix_len, uint32_t left, uint32_t right):
    cdef InteriorNodeC interior_node = InteriorNodeC(prefix_len=prefix_len, suffix_len=suffix_len, left=left, right=right)
    cdef NodeC inner = NodeC(interior_node=interior_node)
    cdef EditTreeNodeC node = EditTreeNodeC(is_interior_node=True, inner=inner)
    return node

cdef inline EditTreeNodeC edit_tree_node_new_replacement(attr_t replacee, attr_t replacement):
    cdef EditTreeNodeC node
    node.is_interior_node = False
    node.inner.replacement_node.replacee = replacee
    node.inner.replacement_node.replacement = replacement
    return node

cdef inline uint64_t edit_tree_node_hash(EditTreeNodeC node):
    cdef InteriorNodeC interior
    cdef ReplacementNodeC replacement

    if node.is_interior_node:
        interior = node.inner.interior_node
        return hash((interior.prefix_len, interior.suffix_len, interior.left, interior.right))
    else:
        replacement = node.inner.replacement_node
        return hash((replacement.replacee, replacement.replacement))

cdef struct LCS:
    int source_begin
    int source_end
    int target_begin
    int target_end

cdef inline lcs_is_empty(lcs: LCS):
    return lcs.source_begin == 0 and lcs.source_end == 0 and lcs.target_begin == 0 and lcs.target_end == 0

cdef class EditTrees:
    cdef vector[EditTreeNodeC] nodes
    cdef unordered_map[hash_t, uint32_t] map
    cdef StringStore strings

    cpdef str apply(self, uint32_t tree, str form)
    cdef _apply(self, uint32_t node, str form_part, list lemma_pieces)
    cdef uint32_t build(self, str form, str lemma)
    cpdef tree_str(self, uint32_t node)
