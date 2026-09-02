from dataclasses import dataclass

import numpy as np


TAU = 0.95
K_MIN = 1
K_MAX = 8
GAMMA = 3.0
BETA = 0.10
BEAM_WIDTH = 250
EPSILON = 1e-12

LABEL_TO_CHAR = {
    i: chr(ord("a") + i)
    for i in range(26)
}
CHAR_TO_LABEL = {
    char: i
    for i, char in LABEL_TO_CHAR.items()
}
TRIE_TERMINAL = "__word__"

def stable_log(value):
    return np.log(np.clip(np.asarray(value, dtype=np.float64), EPSILON, 1.0))

def normalize_alphabetic_posteriors(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError("Posterior stream must be a 2-D matrix.")

    if probabilities.shape[1] < 26:
        raise ValueError("Posterior stream must contain at least 26 classes.")

    alphabetic = np.clip(probabilities[:, :26], EPSILON, None)

    alphabetic /= alphabetic.sum(axis=1, keepdims=True)

    return alphabetic

def adaptive_k_values(probabilities):
    probabilities = normalize_alphabetic_posteriors(probabilities)

    order = np.argsort(-probabilities, axis=1)

    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)

    cumulative = np.cumsum(sorted_probabilities, axis=1)

    candidate_ks = np.arange(K_MIN, K_MAX + 1, dtype=np.int64)

    threshold_hits = (cumulative[:, candidate_ks - 1] >= TAU)

    has_hit = threshold_hits.any(axis=1)
    first_hit = np.argmax(threshold_hits, axis=1)

    selected = np.full(probabilities.shape[0], K_MAX, dtype=np.int64)

    selected[has_hit] = candidate_ks[first_hit[has_hit]]

    return selected

def build_trie(words):
    root = {}

    for word in words:
        node = root

        for character in word:
            node = node.setdefault(character, {})

        node[TRIE_TERMINAL] = word

    return root

def word_log_probability(word, probabilities):
    labels = np.asarray(
        [
            CHAR_TO_LABEL[character]
            for character in word
        ],
        dtype=np.int64,
    )

    rows = np.arange(len(labels), dtype=np.int64)

    return float(stable_log(probabilities[rows, labels]).sum())

def fallback_word(probabilities, lexicon, zipf_frequency):
    return max(lexicon, key=lambda word: (word_log_probability(word, probabilities) + BETA * zipf_frequency[word], zipf_frequency[word], word))

def decode_word_original(posterior_stream, lexicon, zipf_frequency, trie=None):
    probabilities = normalize_alphabetic_posteriors(posterior_stream)

    if trie is None:
        trie = build_trie(lexicon)

    selected_k = adaptive_k_values(probabilities)

    order = np.argsort(-probabilities, axis=1)
    ranks = np.argsort(order, axis=1)

    preferred = [
        set(
            order[position, :selected_k[position]]
        )
        for position in range(len(selected_k))
    ]

    beam = [(0.0, "", trie)]

    for position in range(probabilities.shape[0]):
        expanded = []

        confidence = float(np.max(probabilities[position, :26]))

        for score, prefix, node in beam:
            for character, child in node.items():
                if character == TRIE_TERMINAL:
                    continue

                label = CHAR_TO_LABEL[character]

                edge_score = float(stable_log(probabilities[position, label]))

                if label not in preferred[position]:
                    rank_fraction = (float(ranks[position, label]) / 25.0)

                    edge_score -= (GAMMA * confidence * (1.0 + rank_fraction))

                expanded.append((score + edge_score, prefix + character, child))

        if not expanded:
            beam = []
            break

        expanded.sort(key=lambda item: item[0], reverse=True)

        beam = expanded[:BEAM_WIDTH]

    completed = []

    for score, prefix, node in beam:
        if TRIE_TERMINAL in node:
            word = node[TRIE_TERMINAL]

            completed.append((score + BETA * zipf_frequency[word], word))

    if completed:
        completed.sort(reverse=True)
        return completed[0][1]

    return fallback_word(probabilities, lexicon, zipf_frequency)


@dataclass
class ArrayTrie:
    words: list
    frequencies: np.ndarray
    label_matrix: np.ndarray
    children: np.ndarray
    terminal_word_index: np.ndarray

def build_array_trie(lexicon, zipf_frequency):
    words = sorted(set(lexicon))

    children = [np.full(26, -1, dtype=np.int32)]

    terminal_word_index = [-1]

    for word_index, word in enumerate(words):
        node_index = 0

        for character in word:
            label = CHAR_TO_LABEL[character]

            child_index = int(children[node_index][label])

            if child_index < 0:
                child_index = len(children)

                children[node_index][label] = child_index

                children.append(np.full(26, -1, dtype=np.int32))

                terminal_word_index.append(-1)

            node_index = child_index

        terminal_word_index[node_index] = word_index

    label_matrix = np.asarray(
        [
            [
                CHAR_TO_LABEL[character]
                for character in word
            ]
            for word in words
        ],
        dtype=np.int16,
    )

    frequencies = np.asarray(
        [
            zipf_frequency[word]
            for word in words
        ],
        dtype=np.float64,
    )

    return ArrayTrie(words=words, frequencies=frequencies, label_matrix=label_matrix, children=np.vstack(children).astype(np.int32, copy=False), 
                     terminal_word_index=np.asarray(terminal_word_index, dtype=np.int32))

def optimized_fallback(probabilities, array_trie):
    rows = np.arange(probabilities.shape[0])[:, None]

    emissions = stable_log(probabilities[rows, array_trie.label_matrix.T]).sum(axis=0)

    scores = (emissions + BETA * array_trie.frequencies)

    best_score = float(np.max(scores))

    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))

    if len(tied) == 1:
        return array_trie.words[int(tied[0])]

    # Same frequency-and-word tie rule as the fallback.
    best_index = max(tied, key=lambda index: (array_trie.frequencies[int(index)], array_trie.words[int(index)]))

    return array_trie.words[int(best_index)]

def decode_word_optimized(posterior_stream, array_trie):
    probabilities = normalize_alphabetic_posteriors(posterior_stream)

    selected_k = adaptive_k_values(probabilities)

    order = np.argsort(-probabilities, axis=1)
    ranks = np.argsort(order, axis=1)

    log_probability = stable_log(probabilities[:, :26])

    confidence = np.max(probabilities[:, :26], axis=1)

    outside = (ranks >= selected_k[:, None])

    rank_fraction = (ranks.astype(np.float64) / 25.0)

    edge_scores = (log_probability - outside * GAMMA * confidence[:, None] * (1.0 + rank_fraction))

    beam_nodes = np.asarray([0], dtype=np.int32)

    beam_scores = np.asarray([0.0], dtype=np.float64)

    for position in range(probabilities.shape[0]):
        child_matrix = (array_trie.children[beam_nodes])

        valid_mask = (child_matrix >= 0)

        parent_rows, labels = np.nonzero(valid_mask)

        if len(parent_rows) == 0:
            beam_nodes = np.empty(0, dtype=np.int32)
            beam_scores = np.empty(0, dtype=np.float64)
            break

        candidate_nodes = (child_matrix[parent_rows, labels])

        candidate_scores = (beam_scores[parent_rows] + edge_scores[position, labels])

        # Preserve generation order when scores tie.
        generation_order = np.arange(len(candidate_scores), dtype=np.int64)

        ranked = np.lexsort((generation_order, -candidate_scores))

        keep = ranked[:BEAM_WIDTH]

        beam_nodes = candidate_nodes[keep]

        beam_scores = candidate_scores[keep]

    completed = []

    if len(beam_nodes):
        terminal_indices = (array_trie.terminal_word_index[beam_nodes])

        for score, word_index in zip(beam_scores, terminal_indices):
            if int(word_index) < 0:
                continue

            word_index = int(word_index)

            completed.append((float(score) + BETA * array_trie.frequencies[word_index], array_trie.words[word_index]))

    if completed:
        return max(completed)[1]

    return optimized_fallback(probabilities, array_trie)


# Use:
#
# trie = build_trie(words)
#
# word_original = decode_word_original(posterior_stream, words, zipf_scores, trie=trie)
#
# array_trie = build_array_trie(words, zipf_scores)
#
# word_optimized = decode_word_optimized(posterior_stream, array_trie)
#
# assert word_original == word_optimized
