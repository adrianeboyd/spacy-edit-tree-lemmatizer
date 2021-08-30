from typing import List, Optional

from spacy import registry
from spacy.tokens import Doc
from thinc.initializers import zero_init
from thinc.layers import chain, Softmax, with_array
from thinc.model import Model
from thinc.types import Floats2d


@registry.architectures("edit_tree_model.v1")
def build_edit_tree_model(
    tok2vec: Model[List[Doc], List[Floats2d]], nO: Optional[int] = None
) -> Model[List[Doc], List[Floats2d]]:
    t2v_width = tok2vec.get_dim("nO") if tok2vec.has_dim("nO") else None
    output_layer = Softmax(nO, t2v_width, init_W=zero_init)
    softmax = with_array(output_layer)
    model = chain(tok2vec, softmax)
    model.set_ref("tok2vec", tok2vec)
    model.set_ref("softmax", output_layer)
    model.set_ref("output_layer", output_layer)
    return model