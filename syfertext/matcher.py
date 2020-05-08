from .doc import Doc
import re

# TODO: Change name to Matcher! Cause it ain't simple any more.
class SimpleMatcher:
    """Match sequences of tokens, based on pattern rules.
    NOTE: ASSUMPTION IS THAT ALL TOKEN ATTRIBUTES ARE STORED IN `token._` (notice the underscore)
    """

    def __init__(self, vocab):
        """Create the Matcher

        Args:
            vocab (Vocab): The vocabulary object which must be shared with the documents
                matcher will operate on.
        """

        self._patterns = {}
        self._callbacks = {}
        self.vocab = vocab

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

        A pattern consists of one or more `token_specs`, where a `token_spec`
        is a dictionary mapping attribute IDs to values.

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

        TODO: Support quantifier operator.

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
        self._patterns[key] = self._preprocess_patterns(patterns)
        self._callbacks[key] = on_match

    def get(self, key):
        """ Retrieve the pattern stored for a key.

        Args:
            key (unicode): The key of the patterns to retrieve.

        TODO: Convert preprocessed patterns back to dictionary?
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

    def __call__(self, doc_or_span):
        """Find all token sequences matching the supplied pattern.

        Doc must be parsed and it's tokens must have the default attributes.

        Args:
            doc_or_span (Doc or Span): The document to match tokens over.

        Returns:
            A list of `(key, start, end)` tuples, describing the matches.
            A match tuple describes a span `doc[start:end]`. The `key` corresponds
            to the `key` of the pattern that has matched.
        """

        # if isinstance(doc_or_span, Doc):
        #     doc = doc_or_span
        #     length = len(doc)
        # elif isinstance(doc_or_span, Span):
        #     doc = doc_or_span.as_doc()
        #     length = len(doc)

        # TODO: Support Other predicates
        # TODO: Do we need to support Quantifiers ?
        # TODO: Support empty match tokens, will be useful for "Username {name}"

        doc = doc_or_span
        matches = []

        if len(doc) == 0:
            # Avoid processing of an empty doc
            return matches

        length = len(doc)

        # A naive way of performing matching.
        # Time complexity O(Number of patterns * Length of Doc * Max_Length of pattern)
        # Note number of patterns includes multiple patterns with the same key.

        # Iterate over all patterns
        for key, patterns in self._patterns.items():

            # Iterate over all patterns with this key
            for pattern in patterns:

                # Iterate over all tokens in doc
                i = 0
                while i < length:  # While loop allows us to skip tokens after we find a match

                    token = doc_or_span[i]

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
                            token = doc_or_span[temp]
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
    def _preprocess_patterns(patterns):
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

            for part in pattern:

                # Assert it's a dict containing a match pattern
                # for exactly one token
                assert isinstance(part, dict)
                assert len(part) == 1

                for key, value in part.items():

                    attr = key.lower()

                    if isinstance(value, dict):
                        # It is a predicate
                        predicate = _get_predicates(attr, value)
                        cur_pattern.append((attr, predicate))
                    else:
                        cur_pattern.append((attr, value))

            processed_patterns.append(cur_pattern)

        return processed_patterns


# End of Matcher class


def _check_match(target, token):
    """
    Args:
        target (tuple): Consists of attribute and it's value to match
        token (Token): Token whose attribute's value should match with target

    Returns:
        True if target pattern matches with token.
    """
    attr, value = target
    if not hasattr(token._, attr):
        return False

    if isinstance(value, str) or isinstance(value, int):
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
    predicate_types = {"REGEX": _RegexPredicate}

    # Current design supports only one predicate
    assert len(value) == 1

    for pred_type, pred_value in value.items():
        # Initialize an appropriate class based on the predicate type
        predicate_class = predicate_types[pred_type]
        return predicate_class(attr, pred_value)


class _RegexPredicate:
    """Matches the token based on regex.
    Can be applied on `TEXT`, `LOWER` and `TAG` attributes
    of token."""

    # TODO: Extend RegexPredicate for TAG

    def __init__(self, attr, value):
        """
        Args:
            attr (unicode): Attribute of token which will be matched.
            value (unicode): Regex pattern to find in self.attr attribute of token
        """
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
