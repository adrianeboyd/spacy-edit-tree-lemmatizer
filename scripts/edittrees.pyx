# cython: infer_types=True, binding=True
from cython.operator cimport dereference as deref
from edittrees cimport lcs_is_empty
from pathlib import Path
from libc.stdint cimport uint32_t
from libc.stdint cimport UINT32_MAX
from libc.string cimport memset
from libcpp.pair cimport pair
from libcpp.vector cimport vector
import spacy.util
from spacy.strings import StringStore
from spacy.typedefs cimport hash_t
from typing import Union

from edittrees cimport EditTrees, EditTreeNodeC


NULL_NODE_ID = UINT32_MAX


cdef LCS find_lcs(unicode source, unicode target):
    """
    Find the longest common subsequence (LCS) between two strings. If there are
    multiple LCSes, only one of them is returned.

    :param source: The first string.
    :param target: The second string.
    :return: The spans of the longest common subsequences.
    """
    cdef Py_ssize_t source_len = len(source)
    cdef Py_ssize_t target_len = len(target)
    cdef size_t longest_align = 0;
    cdef int source_idx, target_idx
    cdef LCS lcs
    cdef Py_UCS4 source_cp, target_cp

    memset(&lcs, 0, sizeof(lcs))

    cdef vector[size_t] prev_aligns = vector[size_t](target_len);
    cdef vector[size_t] cur_aligns = vector[size_t](target_len);

    for (source_idx, source_cp) in enumerate(source):
        for (target_idx, target_cp) in enumerate(target):
            if source_cp == target_cp:
                if source_idx == 0 or target_idx == 0:
                    cur_aligns[target_idx] = 1
                else:
                    cur_aligns[target_idx] = prev_aligns[target_idx - 1] + 1

                if cur_aligns[target_idx] > longest_align:
                    longest_align = cur_aligns[target_idx]
                    lcs.source_begin = source_idx - longest_align + 1
                    lcs.source_end =  source_idx + 1
                    lcs.target_begin = target_idx - longest_align + 1
                    lcs.target_end =  target_idx + 1
            else:
                # No match, we start with a zero-length LCS.
                cur_aligns[target_idx] = 0
        swap(prev_aligns, cur_aligns)

    return lcs



cdef class EditTrees:
    def __init__(self, strings: StringStore):
        self.nodes = vector[EditTreeNodeC]()
        self.strings = strings

    def add(self, form: unicode, lemma: unicode) -> int:
        return self.build(form, lemma)

    cdef uint32_t build(self, unicode form, unicode lemma):
        cdef EditTreeNodeC node
        cdef uint32_t node_id, left, right

        cdef LCS lcs = find_lcs(form, lemma)

        if lcs_is_empty(lcs):
            node = edit_tree_node_new_replacement(self.strings.add(form), self.strings.add(lemma))
        else:
            left = NULL_NODE_ID
            if lcs.source_begin != 0 or lcs.target_begin != 0:
                left = self.build(form[:lcs.source_begin], lemma[:lcs.target_begin])

            right = NULL_NODE_ID
            if lcs.source_end != len(form) or lcs.target_end != len(lemma):
                right = self.build(form[lcs.source_end:], lemma[lcs.target_end:])

            node = edit_tree_node_new_interior(lcs.source_begin, len(form) - lcs.source_end, left, right)

        cdef hash_t hash = edit_tree_node_hash(node)
        cdef unordered_map[hash_t, uint32_t].iterator iter = self.map.find(hash)
        if iter != self.map.end():
            return deref(iter).second

        node_id = self.nodes.size()
        self.nodes.push_back(node)
        self.map.insert(pair[hash_t, uint32_t](hash, node_id))

        return node_id

    cpdef unicode apply(self, uint32_t tree, unicode form):
        lemma_pieces = []
        self._apply(tree, form, lemma_pieces)
        return "".join(lemma_pieces)

    cdef _apply(self, uint32_t tree, unicode form_part, list lemma_pieces):
        cdef EditTreeNodeC node = self.nodes[tree]
        cdef InteriorNodeC interior
        cdef int suffix_start

        if node.is_interior_node:
            interior = node.inner.interior_node
            suffix_start = len(form_part) - interior.suffix_len

            if interior.left != NULL_NODE_ID:
                self._apply(interior.left, form_part[:interior.prefix_len], lemma_pieces)

            lemma_pieces.append(form_part[interior.prefix_len:suffix_start])

            if interior.right != NULL_NODE_ID:
                self._apply(interior.right, form_part[suffix_start:], lemma_pieces)
        else:
            lemma_pieces.append(self.strings[node.inner.replacement_node.replacement])

    cpdef tree_str(self, uint32_t tree):
        cdef EditTreeNodeC node = self.nodes[tree]
        cdef InteriorNodeC interior
        cdef ReplacementNodeC replacement

        if node.is_interior_node:
            interior = node.inner.interior_node

            left = "()"
            if interior.left != NULL_NODE_ID:
                left = self.tree_str(interior.left)

            right = "()"
            if interior.right != NULL_NODE_ID:
                right = self.tree_str(interior.right)

            return f"(i {interior.prefix_len} {interior.suffix_len} {left} {right})"

        replacement = node.inner.replacement_node

        return f"(r '{self.strings[replacement.replacee]}' '{self.strings[replacement.replacement]}')"

    def from_bytes(self, bytes_data: bytes, * ) -> "EditTrees":
        def deserialize_nodes(node_dicts):
            cdef EditTreeNodeC c_node
            for node_dict in node_dicts:
                c_node = node_dict
                self.nodes.push_back(c_node)

        deserializers = {}
        deserializers["nodes"] = lambda n: deserialize_nodes(n)
        spacy.util.from_bytes(bytes_data, deserializers, [])

        self._rebuild_node_map()

        return self

    def to_bytes(self, **kwargs) -> bytes:
        node_dicts = []
        for node in self.nodes:
            node = dict(node)
            if node['is_interior_node']:
                del node['inner']['replacement_node']
            else:
                del node['inner']['interior_node']
            node_dicts.append(node)

        serializers = {}
        serializers["nodes"] = lambda: node_dicts

        return spacy.util.to_bytes(serializers, [])


    def to_disk(self, path: Union[str, Path], **kwargs) -> "EditTrees":
        path = spacy.util.ensure_path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes())

    def from_disk(self, path: Union[str, Path], **kwargs ) -> "EditTrees":
        path = spacy.util.ensure_path(path)
        if path.exists():
            with path.open("rb") as file_:
                data = file_.read()
            return self.from_bytes(data)

        return self

    def size(self):
        return self.nodes.size()

    def _rebuild_node_map(self):
        cdef EditTreeNodeC c_node
        cdef uint32_t node_id
        cdef hash_t node_hash

        self.map.clear()

        for node_id in range(self.nodes.size()):
            c_node = self.nodes[node_id]
            node_hash = edit_tree_node_hash(c_node)
            self.map.insert(pair[hash_t, uint32_t](node_hash, node_id))
