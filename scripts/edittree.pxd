from spacy.typedefs cimport attr_t, len_t
from spacy.vocab cimport Vocab

cdef struct EditTreeNodeC

cdef struct InteriorNodeC:
    len_t prefix_len
    len_t suffix_len
    EditTreeNodeC *left
    EditTreeNodeC *right

cdef struct ReplacementNode:
    attr_t replacee
    attr_t replacement

cdef union NodeC:
    InteriorNodeC interior_node
    ReplacementNode replacement_node

cdef struct EditTreeNodeC:
    NodeC node
    bint is_interior_node
