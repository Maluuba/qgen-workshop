#!/usr/bin/env python

import os

import numpy as np

_word_to_idx = {}
_idx_to_word = []


def _add_word(word):
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"

UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
START_TOKEN = _add_word(START_WORD)
END_TOKEN = _add_word(END_WORD)


def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)


def look_up_token(token):
    return _idx_to_word[token]


embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glove.6B.100d.trimmed.txt')
with open(embeddings_path) as f:
    line = f.readline()
    chunks = line.split(" ")
    dimensions = len(chunks) - 1
    f.seek(0)

    vocab_size = sum(1 for line in f)
    vocab_size += 3
    f.seek(0)

    glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
    glove[UNKNOWN_TOKEN] = np.zeros(dimensions)
    glove[START_TOKEN] = -np.ones(dimensions)
    glove[END_TOKEN] = np.ones(dimensions)

    for line in f:
        chunks = line.split(" ")
        idx = _add_word(chunks[0])
        glove[idx] = [float(chunk) for chunk in chunks[1:]]
        if len(_idx_to_word) >= vocab_size:
            break
