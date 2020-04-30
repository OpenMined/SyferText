from .doc import Doc

# from .span import Span


class SimpleMatcher:
    """Match sequences of tokens, based on pattern rules.
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
        TODO: Support on match callback.

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

    # def has_key(self, key):
    #     """Checks whether the matcher has a rule with the given key.
    #
    #     Args:
    #         key (string or int): The key to check
    #
    #     Returns:
    #         True if the matcher has rule, else False.
    #     """
    #
    #     key = self._normalize_key(key)
    #     return key in self._patterns

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
            for p in patterns:

                # Iterate over all tokens in doc
                i = 0
                while i < length:  # Allows us to skip tokens after we find a match

                    token = doc_or_span[i]

                    ptr = 0
                    temp = i

                    # Try matching this token with the beginning of current pattern
                    while hasattr(token._, p[ptr][0]) and getattr(token._, p[ptr][0]) == p[ptr][1]:

                        ptr += 1
                        temp += 1

                        # All the sub parts of this pattern matched
                        if ptr == len(p):

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
                            [{"LOWER": "hello"}, {"LOWER": "world"}]
                    ]

            processed_patterns = [[("lower", "hello"), ("is_punct", True), ("lower", "world")],
                                        [("lower", "hello"), ("lower", "world")]
                                ]
        """

        processed_patterns = list()

        for pattern in patterns:

            assert isinstance(pattern, list)
            cur_pattern = list()

            for part in pattern:

                assert isinstance(part, dict)
                item = [(key.lower(), value) for key, value in part.items()]

                assert len(item) == 1
                cur_pattern.append(item[0])

            processed_patterns.append(cur_pattern)

        return processed_patterns
