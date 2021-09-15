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

from edittrees cimport EditTrees, EditTreeC


NULL_TREE_ID = UINT32_MAX


cdef LCS find_lcs(str source, str target):
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

                # Check if this is the longest alignment and replace previous
                # best alignment when this is the case.
                if cur_aligns[target_idx] > longest_align:
                    longest_align = cur_aligns[target_idx]
                    lcs.source_begin = source_idx - longest_align + 1
                    lcs.source_end =  source_idx + 1
                    lcs.target_begin = target_idx - longest_align + 1
                    lcs.target_end =  target_idx + 1
            else:
                # No match, we start with a zero-length alignment.
                cur_aligns[target_idx] = 0
        swap(prev_aligns, cur_aligns)

    return lcs


cdef class EditTrees:
    def __init__(self, strings: StringStore):
        self.strings = strings

    def add(self, form: str, lemma: str) -> int:
        return self.build(form, lemma)

    cdef uint32_t build(self, str form, str lemma):
        cdef EditTreeC tree
        cdef uint32_t tree_id, prefix_tree, suffix_tree

        cdef LCS lcs = find_lcs(form, lemma)

        if lcs_is_empty(lcs):
            tree = edittree_new_replacement(self.strings.add(form), self.strings.add(lemma))
        else:
            # If we have a non-empty LCS, such as "gooi" in "ge[gooi]d" and "[gooi]en",
            # create edit trees for the prefix pair ("ge"/"") and the suffix pair ("d"/"en").
            prefix_tree = NULL_TREE_ID
            if lcs.source_begin != 0 or lcs.target_begin != 0:
                prefix_tree = self.build(form[:lcs.source_begin], lemma[:lcs.target_begin])

            suffix_tree = NULL_TREE_ID
            if lcs.source_end != len(form) or lcs.target_end != len(lemma):
                suffix_tree = self.build(form[lcs.source_end:], lemma[lcs.target_end:])

            tree = edittree_new_interior(lcs.source_begin, len(form) - lcs.source_end, prefix_tree, suffix_tree)

        # If this tree has been constructed before, return its identifier.
        cdef hash_t hash = edittree_hash(tree)
        cdef unordered_map[hash_t, uint32_t].iterator iter = self.map.find(hash)
        if iter != self.map.end():
            return deref(iter).second

        #  The tree hasn't been seen before, store it.
        tree_id = self.trees.size()
        self.trees.push_back(tree)
        self.map.insert(pair[hash_t, uint32_t](hash, tree_id))

        return tree_id

    cpdef str apply(self, uint32_t tree_id, str form):
        lemma_pieces = []
        try:
            self._apply(tree_id, form, lemma_pieces)
        except ValueError:
            return None
        return "".join(lemma_pieces)

    cdef _apply(self, uint32_t tree_id, str form_part, list lemma_pieces):
        cdef EditTreeC tree = self.trees[tree_id]
        cdef InteriorNodeC interior
        cdef int suffix_start

        if tree.is_interior_node:
            interior = tree.inner.interior_node
            suffix_start = len(form_part) - interior.suffix_len

            if interior.prefix_tree != NULL_TREE_ID:
                self._apply(interior.prefix_tree, form_part[:interior.prefix_len], lemma_pieces)

            lemma_pieces.append(form_part[interior.prefix_len:suffix_start])

            if interior.suffix_tree != NULL_TREE_ID:
                self._apply(interior.suffix_tree, form_part[suffix_start:], lemma_pieces)
        else:
            if form_part == self.strings[tree.inner.replacement_node.replacee]:
                lemma_pieces.append(self.strings[tree.inner.replacement_node.replacement])
            else:
                raise ValueError("Edit tree cannot be applied to form")

    cpdef tree_str(self, uint32_t tree_id):
        if tree_id >= self.trees.size():
            raise ValueError("Unknown edit tree")

        cdef EditTreeC tree = self.trees[tree_id]
        cdef ReplacementNodeC replacement

        if not tree.is_interior_node:
            replacement = tree.inner.replacement_node
            return f"(r '{self.strings[replacement.replacee]}' '{self.strings[replacement.replacement]}')"

        cdef InteriorNodeC interior = tree.inner.interior_node

        left = "()"
        if interior.prefix_tree != NULL_TREE_ID:
            left = self.tree_str(interior.prefix_tree)

        right = "()"
        if interior.suffix_tree != NULL_TREE_ID:
            right = self.tree_str(interior.suffix_tree)

        return f"(i {interior.prefix_len} {interior.suffix_len} {left} {right})"


    def from_bytes(self, bytes_data: bytes, * ) -> "EditTrees":
        def deserialize_trees(tree_dicts):
            cdef EditTreeC c_tree
            for tree_dict in tree_dicts:
                c_tree = tree_dict
                self.trees.push_back(c_tree)

        deserializers = {}
        deserializers["trees"] = lambda n: deserialize_trees(n)
        spacy.util.from_bytes(bytes_data, deserializers, [])

        self._rebuild_tree_map()

        return self

    def to_bytes(self, **kwargs) -> bytes:
        tree_dicts = []
        for tree in self.trees:
            tree = dict(tree)
            if tree['is_interior_node']:
                del tree['inner']['replacement_node']
            else:
                del tree['inner']['interior_node']
            tree_dicts.append(tree)

        serializers = {}
        serializers["trees"] = lambda: tree_dicts

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
        return self.trees.size()

    def _rebuild_tree_map(self):
        cdef EditTreeC c_tree
        cdef uint32_t tree_id
        cdef hash_t tree_hash

        self.map.clear()

        for tree_id in range(self.trees.size()):
            c_tree = self.trees[tree_id]
            tree_hash = edittree_hash(c_tree)
            self.map.insert(pair[hash_t, uint32_t](tree_hash, tree_id))
