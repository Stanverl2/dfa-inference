import numpy as np

from dfa import Node, generate_prefix_tree, render_dfa
from dfa_generation import generate_dfa


def main():
    """
    Example usages of the code.

    :return:
    """
    nodes: list[Node] = generate_dfa(4, ())
    root: Node = nodes[0]

    assert root.id == 0

    words = set()
    while len(words) < 15:
        words.add(generate_random_word())

    words_in_language: set[str] = set()
    words_not_in_language: set[str] = set()

    for word in words:
        if root.accepts(word):
            words_in_language.add(word)
        else:
            words_not_in_language.add(word)

    prefix_tree: Node = generate_prefix_tree(words_in_language, words_not_in_language)

    render_dfa(prefix_tree)


def generate_random_word(length=5, alphabet=None) -> str:
    if alphabet is None:
        alphabet = ['a', 'b']

    return ''.join(np.array(alphabet)[np.random.randint(0, len(alphabet), size=length)])


if __name__ == '__main__':
    main()
