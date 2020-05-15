from syfertext.doc import Doc
from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde

import re
from typing import Dict
from typing import List
from typing import Union


class Matcher:
    """Match sequences of tokens, based on pattern rules.
    NOTE: ASSUMPTION IS THAT ALL TOKEN ATTRIBUTES ARE STORED IN `token._` (notice the underscore)
    """

    def __init__(self, vocab):
        """Create the Matcher

        Args:
            vocab (Vocab): The vocabulary object which must be shared with the documents
                matcher will operate on.
        """

        self.vocab = vocab

        # Initialize empty dicts
        self._patterns = {}
        self._preprocessed_patterns = {}
        self._callbacks = {}

    def __len__(self):
        """Get the number of rules added ot the matcher."""
        return len(self._patterns)

    def __contains__(self, key):
        """Check whether the matcher contains rules for a match ID.

        Args:
            key (unicode): The match ID.

        Returns:
            True if the matcher contains the rules for this match ID.
        """

        return self._normalize_key(key) in self._patterns

    def add(self, key, patterns, on_match=None):
        """Add a match-rule to the matcher. A match-rule consists of: an ID
        key and one or more patterns.

        A pattern consists of one or more `token_specs` (token specifications),
        where a `token_spec` is a dictionary mapping attribute IDs to values.

        Example:
            1. Adding a single pattern to matcher.

                pattern = [{"LOWER": "facebook"}, {"LEMMA": "be"}, {"POS": "ADV"},
                           {"POS": "ADJ"}]

                matcher.add("facebook", [pattern])

            2. Adding multiple patterns with the same key

            matcher.add("HelloWorld",
                        [[{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}],
                            [{"LOWER": "hello"}, {"LOWER": "world"}]
                        ]
                    )

            Note:
                 Each dictionary is a single token

        TODO: Avoid the necessity of pattern being a Lists of Lists.

        Args:
            key (unicode): The match ID.
            patterns (List): The patterns to add for the given key.
            on_match (Callable): Optional callback executed on match.

        Returns:
        """

        # Asserting on match is a callable object
        if on_match is not None and not hasattr(on_match, "__call__"):
            raise TypeError(f"`on_match` {on_match} is not callable")

        key = self._normalize_key(key)
        self._patterns[key] = patterns
        self._preprocessed_patterns[key] = self._process_patterns(patterns)
        self._callbacks[key] = on_match

    def get(self, key):
        """ Retrieve the pattern stored for a key.

        Args:
            key (unicode): The key of the patterns to retrieve.

        Returns:
            The rule, as an (on_match, patterns) tuple.
        """

        key = self._normalize_key(key)
        return self._patterns[key]

    def remove(self, key):
        """Remove a rule from the matcher.

        Args:
            key (unicode): The ID of the match rule.
        """

        key = self._normalize_key(key)
        try:
            del self._patterns[key]
        except KeyError:
            pass

    def __call__(self, doc):
        """Find all token sequences matching the supplied pattern.

        Doc must be parsed and it's tokens must have the default attributes.

        Args:
            doc (Doc): The document to match tokens over.

        Returns:
            A list of `(key, start, end)` tuples, describing the matches.
            A match tuple describes a span `doc[start:end]`. The `key` corresponds
            to the `key` of the pattern that has matched.
        """

        # TODO: Add support for Span.

        # if isinstance(doc, Doc):
        #     doc = doc
        #     length = len(doc)
        # elif isinstance(doc, Span):
        #     doc = doc.as_doc()
        #     length = len(doc)

        # TODO: Support Other predicates
        # TODO: Do we need to support Quantifiers ?
        # TODO: Support empty match tokens, will be useful for "Username {name}"

        # doc = doc
        matches = []

        if len(doc) == 0:
            # Avoid processing of an empty doc
            return matches

        length = len(doc)

        # A naive way of performing matching.
        # Time complexity O(Number of patterns * Length of Doc * Max_Length of pattern)
        # Note number of patterns includes multiple patterns with the same key.

        # Iterate over all patterns
        for key, patterns in self._preprocessed_patterns.items():

            # Iterate over all patterns with this key
            for pattern in patterns:

                # Iterate over all tokens in doc
                i = 0
                while i < length:  # While loop allows us to skip tokens after we find a match

                    token = doc[i]

                    ptr = 0
                    temp = i

                    # Try matching this token with the beginning of current pattern
                    while _check_match(pattern[ptr], token):

                        ptr += 1
                        temp += 1

                        # All the sub parts of this pattern matched
                        if ptr == len(pattern):

                            # Add this match to the output
                            matches.append((key, i, temp))

                            # Code continues from the next token
                            # of the last token in the match
                            i = temp - 1  # Minus one cause at the end there is a +1
                            break

                        # If sub parts to be matched are still in pattern
                        # Try matching the next token to the next sub part of pattern
                        if temp < length:
                            token = doc[temp]
                        else:
                            break

                    i += 1

        # Execute callback on the matches
        for i, (key, start, end) in enumerate(matches):

            # Get the callback function
            on_match = self._callbacks.get(key, None)

            if on_match is not None:
                on_match(self, doc, i, matches)

        return matches

    def _normalize_key(self, key):
        """Returns the hash value of the string. Saves memory by storing
        hash values instead of texts.

        Args:
            key (string or unicode): The key to normalize.

        Returns:
            Hash value of the string, if key is a string. Else, returns the
            hash value itself.
        """

        if isinstance(key, str):
            return self.vocab.store[key]
        else:
            return key

    @staticmethod
    def _process_patterns(patterns):
        """Saves list of list of dicts as list of list of tuples.

        Example:
            patterns = [[{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}],
                            [{"LOWER": "hello"}, {"lower': {"REGEX": "^[Ii](ndia)"}}]
                    ]

            processed_patterns = [[("lower", "hello"), ("is_punct", True), ("lower", "world")],
                                    [("lower", "hello"), ("lower", _RegexPredicate(attr="lower", value="^[Ii](ndia)")]
                                ]
        """

        processed_patterns = list()

        for pattern in patterns:

            assert isinstance(pattern, list)
            cur_pattern = list()

            for token_spec in pattern:

                # Assert it's a dict containing a match pattern
                # for exactly one token
                assert isinstance(token_spec, dict)
                assert len(token_spec) == 1

                for attr, value in token_spec.items():

                    attr = attr.lower()

                    if isinstance(value, dict):
                        # It is a predicate
                        predicate = _get_predicates(attr, value)
                        cur_pattern.append((attr, predicate))
                    else:
                        cur_pattern.append((attr, value))

            processed_patterns.append(cur_pattern)

        return processed_patterns

    @staticmethod
    def simplify(worker: BaseWorker, matcher: "Matcher"):
        """Simplifies a Matcher object.

        Args:
            worker (BaseWorker): The worker on which the simplify operation is carried out.
            matcher (Matcher): the Matcher object to simplify.

        Returns:
            (tuple): The simplified Matcher object.
        """

        # Simplify the object properties
        # TODO: HOW TO SIMPLIFY VOCAB OBJECT ?
        # IT REQUIRES ACCESS TO THE LANGUAGE OBJECT"S VOCAB
        vocab = serde._simplify(worker, matcher.vocab)
        patterns = serde._simplify(worker, matcher._patterns)
        callbacks = serde._simplify(worker, matcher._callbacks)

        return (vocab, patterns, callbacks)

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Takes a simplified Matcher object, details it
           and returns a Matcher object.

        Args:
            worker (BaseWorker): The worker on which the detail operation is carried out.
            simple_obj (tuple): The simplified SubPipeline object.

        Returns:
            (Matcher): The Matcher object.
        """

        # Unpack the simplified object
        vocab, patterns, callbacks = simple_obj

        # Detail each property
        vocab = serde._detail(worker, vocab)
        patterns = serde._detail(worker, patterns)
        callbacks = serde._detail(worker, callbacks)

        # Instantiate a Matcher object
        matcher = Matcher(vocab=vocab)

        # Add patterns to matcher
        for key, patterns_ in patterns.items():
            matcher.add(key, patterns_, callbacks.get(key, None))

        return matcher


# End of Matcher class


def _check_match(token_spec, token):
    """
    Args:
        token_spec (tuple): Consists of attribute and it's value to match.
        token (Token): Token whose attribute's value should match with target

    Returns:
        True if target pattern matches with token.
    """

    attr, value = token_spec
    if not hasattr(token._, attr):
        return False

    if isinstance(value, str) or isinstance(value, bool):
        # Perform normal comparison
        return getattr(token._, attr) == value
    else:
        # It is a predicate
        assert callable(value)  # Assert it has `__call__()`
        return value(token)


def _get_predicates(attr, value):
    """
    Args:
        attr (unicode): Attribute of token to which Predicate will compare it's value
        value (dict): Contains keys as predicate_type (Eg. "REGEX") and value as corresponding
            value to compare
    Returns:
        Reference to new initialized predicate class, depending upon predicate type.
    """

    predicate_types = {
        "REGEX": _RegexPredicate,
        "IN": _SetMemberPredicate,
        "NOT_IN": _SetMemberPredicate,
    }

    # Current design supports only one predicate
    assert len(value) == 1

    for pred_type, pred_value in value.items():
        # Initialize an appropriate class based on the predicate type
        predicate_class = predicate_types[pred_type]
        return predicate_class(attr, pred_type, pred_value)


class _RegexPredicate:
    """Matches the token based on regex.
    Can be applied on `TEXT`, `LOWER` and `TAG` attributes
    of token."""

    # TODO: Extend RegexPredicate for TAG

    def __init__(self, attr, predicate, value):
        """
        Args:
            attr (unicode): Attribute of token which will be matched.
            predicate (unicode): The type of predicate
            value (unicode): Regex pattern to find in self.attr attribute of token
        """

        self.predicate = predicate
        assert self.predicate is "REGEX"

        self.attr = attr
        self.value = re.compile(value)

    def __call__(self, token):
        """
        Args:
            token: Token which needs to be matched

        Returns:
            True if the `self.value` regex pattern finds matches
            with the `self.attr` attribute of token.
        """

        attr_value = getattr(token._, self.attr)
        return bool(self.value.search(attr_value))

    def __repr__(self):
        return f"_RegexPredicate ({self.attr}, {self.value})"


class _SetMemberPredicate:
    """Matches token attribute value against a set of values.
    """

    # Todo: Optimize memory storage by storing hashes of strings

    def __init__(self, attr, predicate, values):
        """
        Args:
            attr (unicode): Attribute of token which will be matched.
            predicate (unicode): Type of predicate
            values (list): Dictionary containing key in ["IN", "NOT_IN"] and list of values to check against
                When key is "IN" then if value of token's attr attribute is in the list, then
                it is a successful match else not a unsuccessful match
                When key is "NOT_IN" then if value of token's attr attribute is not in the list,
                then it is a successful match else unsuccessful match
        """

        self.attr = attr
        self.predicate = predicate
        assert self.predicate in [
            "IN",
            "NOT_IN",
        ], "Please pass only `IN` or `NOT_IN` against a list of values for the attribute"

        self.values = set(values)  # Storing set of values

    def __call__(self, token):
        """
        Args:
            token: Token which needs to be matched

        Returns:
            True if a successful match else False
        """

        attr_value = getattr(token._, self.attr)
        if self.predicate == "IN":
            return attr_value in self.values

        elif self.predicate == "NOT_IN":
            return attr_value not in self.values
