<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-edit-tree-lemmatizer

An experimental edit tree lemmatizer for spaCy v3.2+.

## Install

Until spaCy v3.2.0 is released, install spaCy from `develop` and this package
without build isolation:

```bash
pip install https://github.com/explosion/spacy/archive/develop.zip
pip install -r requirements.txt
pip install --no-build-isolation --editable .
```

Install from source:

```bash
pip install -U pip setuptools wheel
pip install .
```

Or from the repo URL:

```bash
pip install -U pip setuptools wheel
pip install https://github.com/explosion/spacy-edit-tree-lemmatizer/archive/main.zip
```

## Usage

Once this package is installed, the edit tree lemmatizer is registered as a
spaCy component factory, so you can specify it like this in your config:

```ini
[components.edit_tree_lemmatizer]
factory = "edit_tree_lemmatizer"
```

Or start from a blank model in python:

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("edit_tree_lemmatizer")
```

## Demo Project

See training examples in the [`demo project`](project).
