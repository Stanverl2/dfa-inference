import unittest

from dfa import generate_prefix_tree


class TestDfaMethods(unittest.TestCase):

    def test_prefix_tree_word_not_from_alphabet(self):
        """
        Tests whether a warning is raised when a word is encountered with letters not in the
        alphabet ({a,b}).

        :return: Assertion checking if a warning is raised.
        """
        words_in_language: set[str] = {"ab", "aab", "bab", "aaab", "babab", "abbab"}
        words_not_in_language: set[str] = {"a", "b", "cab", "aa", "abba", "babba"}

        with self.assertWarns(RuntimeWarning):
            generate_prefix_tree(words_in_language, words_not_in_language)

    def test_prefix_tree_word_in_both_sets(self):
        """
        Tests whether an error is raised when the positive set and the negative set contain the same
        word.

        :return: Assertion checking if an error is raised.
        """
        words_in_language: set[str] = {"ab", "aab", "bab", "aaab", "babab", "abbab"}
        words_not_in_language: set[str] = {"a", "b", "bab", "aa", "abba", "babba"}

        with self.assertRaises(RuntimeError):
            generate_prefix_tree(words_in_language, words_not_in_language)
