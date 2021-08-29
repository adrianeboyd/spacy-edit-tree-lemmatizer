from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
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

cdef inline EditTreeNodeC *edit_tree_node_new_interior(len_t prefix_len, len_t suffix_len, EditTreeNodeC *left, EditTreeNodeC *right):
    cdef EditTreeNodeC *node = <EditTreeNodeC *>PyMem_Malloc(sizeof(EditTreeNodeC))
    node.is_interior_node = True
    node.node.interior_node.prefix_len = prefix_len
    node.node.interior_node.suffix_len = suffix_len
    node.node.interior_node.left = left
    node.node.interior_node.right = right
    return node

cdef inline EditTreeNodeC *edit_tree_node_new_replacement(attr_t replacee, attr_t replacement):
    cdef EditTreeNodeC *node = <EditTreeNodeC *> PyMem_Malloc(sizeof(EditTreeNodeC))
    node.is_interior_node = False
    node.node.replacement_node.replacee = replacee
    node.node.replacement_node.replacement = replacement
    return node
