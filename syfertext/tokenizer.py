from .doc import Doc
from .vocab import Vocab

from .punctuations import prefix_re, infix_re, suffix_re
from .token_exception import TOKENIZER_EXCEPTIONS
from .underscore import Underscore
from .utils import hash_string


import re
from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.string import String

import pickle
from collections import defaultdict
from typing import List, Union, Tuple, Match, DefaultDict


class TokenMeta(object):
    """This class holds some meta data about a token from the text held by a Doc object.
       This allows to create a Token object when needed.
    """

    def __init__(
        self, hash_key: int, start_pos: int, end_pos: int, space_after: bool, is_space: bool
    ):
        """Initializes a TokenMeta object

        Args:
            hash_key(int): hash value of the string stored by the Token object
            start_pos (int): The start index of the token in the Doc text.
            end_pos (int): The end index of the token in the Doc text (the end index is
                part of the token).
            space_after (bool): Whether the token is followed by a single white
                space (True) or not (False).
            is_space (bool): Whether the token itself is composed of only white
                spaces (True) or not (false).

        """

        # stores the hash of the hash of the string
        self.orth = hash_key

        self.start_pos = start_pos
        self.end_pos = end_pos
        self.space_after = space_after
        self.is_space = is_space

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()


class Tokenizer(AbstractObject):
    def __init__(
            self,
            model_name: str = None,
            exceptions=TOKENIZER_EXCEPTIONS,
            prefix_search=prefix_re.search,
            suffix_search=suffix_re.search,
            infix_finditer=infix_re.finditer,
    ):
        """Initializes the `Tokenizer` object
           
        Args:
            model_name: The name of the language model to which this
                tokenizer belongs.
            exceptions: Exception cases for the tokenizer.
                Example: "e.g.", "Jr." 
            prefix_search: A function matching the signature of
                `re.compile(string).search` to match prefixes.
                Example: "@username" : "@" is a prefix
            suffix_search: A function matching the signature of
                `re.compile(string).search` to match sufixes.
                Example: "Oh!" : "!" is a suffix
            infix_finditer: A function matching the signature of
                `re.compile(string).finditer` to match infixes.
                Example: "Hell-o" : "-" is an infix
        """

        self.prefix_search = prefix_search
        self.suffix_search = suffix_search
        self.infix_finditer = infix_finditer

        if exceptions:
            self.exceptions = exceptions
        else:
            self.exceptions = {}

            
        self.model_name = model_name
        
        # Create a vocab only if the model name is known
        # The model name might not be know at initialization.
        # This happens when the tokenizer is inialized by
        # The user using `nlp.set_tokenizer(Tokenizer())` where the user
        # is not required to explicitely pass the name of the
        # language model to the Tokenizer constructor for
        # convenience.
        # The language model name will be add later to the
        # tokenizer object when the pipeline is created in
        # Subpipeline.load_template().
        if model_name:
            self.vocab = Vocab(model_name=model_name)


    def __call__(self, text: Union[String, str]):
        """The real tokenization procedure takes place here.
        As in the spaCy library. This is not exactly equivalent to 
        text.split(' '). Because tokens can be white spaces if two or
        more consecutive white spaces are found. Also, this tokenizer
        also takes affixes and exception cases into account.

        Example:
            'I love apples' gives three tokens: 'I', 'love', 'apples'
            'I  love apples ' gives four tokens: 'I', ' ', 'love', 'apples'
            ' I love ' gives three tokens: ' ', 'I', 'love' (yes a single white space
            at the beginning is considered a token)
            'I love-apples' gives 4 tokens: 'I', 'love', '-', 'apples'(infix is 
            tokenized seprately)
        Tokenizing this way helps reconstructing the original string
        without loss of white spaces.
        I think that reconstructing the original string might be a good way
        to blindly verify the sanity of the blind tokenization process.

        Args:
           text (Syft String or str) : The text to be tokenized

        """

        # Create a document that will hold meta data of tokens
        # By meta data I mean the start and end positions of each token
        # in the original text, if the token is followed by a white space,
        # if the token itself is composed of white spaces or not, etc ...

        # I do not assign the Doc here any owner, this will
        # be done by the SupPipeline object that operates
        # this tokenizer.
        doc = Doc(self.vocab)

        # The number of characters in the text
        text_size = len(text)

        # Return empty doc for empty strings("")
        if text_size == 0:
            return doc

        # Initialize a pointer to the position of the first character of 'text'
        pos = 0

        # This is a flag to indicate whether the character we are comparing
        # to is a white space or not
        is_space = text[0].isspace()

        # Start tokenization
        for i, char in enumerate(text):

            # We are looking for a character that is the opposite of 'is_space'
            # if 'is_space' is True, then we want to find a character that is
            # not a space. and vice versa. This event marks the end of a token.
            is_current_space = char.isspace()
            if is_current_space != is_space:
                # Create the TokenMeta object that can be later used to retrieve the token
                # from the text
                token_meta = TokenMeta(
                    # get hash key for string stored in the TokenMeta object, where string is
                    # substring of text from start_pos == pos to end_pos + 1 == (i - 1) + 1
                    # Note: If the store doesn't contain string, then it is added to store
                    # and the corresponding key is returned back
                    hash_key=self.vocab.store[str(text[pos : (i - 1) + 1])],
                    start_pos=pos,
                    end_pos=i - 1,
                    space_after=is_current_space,
                    is_space=is_space,
                )

                # if is_space is True that means detected token is composed of only whitespaces
                # so we dont need to check for prefix, infixes etc.
                if is_space:

                    # Append the token to the document
                    doc.container.append(token_meta)
                else:

                    # Process substring for prefix, infix, suffix and exception cases
                    span = str(text[pos:i])

                    doc = self._tokenize(span, token_meta, doc)

                # Adjust the position 'pos' against which
                # we compare the currently visited chararater
                if is_current_space:
                    pos = i + 1
                else:
                    pos = i

                # Update the character type of which we are searching
                # the opposite (space vs. not space).
                # prevent 'pos' from being out of bound
                if pos < text_size:
                    is_space = text[pos].isspace()

            # Create the last token if the end of the string is reached
            if i == text_size - 1 and pos <= i:

                # Create the TokenMeta object that can be later used to retrieve the token
                # from the text
                token_meta = TokenMeta(
                    # hash key for string stored in the TokenMeta object, where string is
                    # substring of text from start_pos == pos to end_pos == None
                    # Note: If the store doesn't contain string, then it is added to store
                    # and the corresponding key is returned back
                    hash_key=self.vocab.store[str(text[pos:])],
                    start_pos=pos,
                    end_pos=None,  # text[pos:None] ~ text[pos:]
                    space_after=is_current_space,
                    is_space=is_space,
                )

                # if is_space is True that means detected token is composed of only whitespaces
                # so we dont need to check for prefix, infixes etc.
                if is_space:

                    # Append the token to the document
                    doc.container.append(token_meta)
                else:

                    # Process substring for prefix, infix, suffix and exception cases
                    span = str(text[pos:None])
                    doc = self._tokenize(span, token_meta, doc)

        return doc

    def _tokenize(self, substring: str, token_meta: TokenMeta, doc: Doc) -> Doc:
        """ Tokenize each substring formed after splitting affixes and processing 
            exceptions. Returns Doc object.

        Args:
            substring: The substring to tokenize.
            token_meta: The TokenMeta object of original substring
                before splitting affixes and exceptions.
            doc: Document object. 

        Returns:    
            doc: Document with all the TokenMeta objects of every token after splitting 
                affixes and exceptions.
        """

        # Start position of substring in text to be tokenized.
        pos = token_meta.start_pos

        # If there is trailing space after the substring in text.
        space_after = token_meta.space_after

        # Get the remaining substring, it's start pos relative to original text,
        # affixes containing list of TokenMeta for each type affix and
        # list of TokenMeta of exceptions after splitting the affixes.
        substring, pos, affixes, exception_tokens = self._split_affixes(
            substring=substring, start_pos=pos
        )

        # Attach all the `TokenMeta` objects formed as result of splitting
        # the affixes and exception cases in the doc container.
        doc = self._attach_tokens(
            doc=doc,
            substring=substring,
            start_pos=pos,
            space_after=space_after,
            affixes=affixes,
            exception_tokens=exception_tokens,
        )

        return doc

    def _split_affixes(
        self, substring: str, start_pos: int
    ) -> Tuple[str, int, DefaultDict, List[TokenMeta]]:
        """Process substring for tokenizing prefixes, infixes, suffixes and exceptions.

        Args:
            substring: The substring to tokenize.
            start_pos: A pointer to the start position of the substring in the text.

        Returns:    
            substring: The substring to tokenize.
            start_pos: A pointer to the start position of the substring in the text.
            affixes: Dict holding TokenMeta lists of each affix 
                types as a result of splitting affixes
            exception_tokens: The list of exception tokens TokenMeta objects.
        """

        suffixes = []
        prefixes = []
        infixes = []
        exception_tokens = []
        pos = start_pos
        end_pos = pos

        next_affix = ["prefix", "suffix"]

        # Dict holding TokenMeta lists of each affix types(prefix, suffix, infix)
        affixes = defaultdict(list)

        # The first element in the `next_affix` list is 'prefix'.
        # since we should start by finding prefixes, we fix the
        # index i = 0.
        # Start by finding prefixes.
        i = 0

        # Holds the last value of `i` at the moment when either a prefix or a suffix is matched.
        # If the difference between `i` and l`ast_i` is greater than 2, it means that neither a
        # prefix nor a suffix is found.
        last_i = 0

        # In each iteration, `substring` is searched for a prefix first, then for a suffix, and thus
        # alternatively. The loop terminates when an exception substring is encountered,
        # or when the substring is not updated for 2 continuous iterations, i.e, when
        # neither a prefix nor a suffix is matched in the substring.
        while i - last_i <= 2:

            if substring in self.exceptions:
                # Get a list of exception  `TokenMeta` objects to be added in the Doc container
                exception_tokens, substring = self._get_exception_token_metas(substring, pos)

                break

            # Get affix type (prefix or suffix)
            affix_type = next_affix[i % 2]

            affix_finder = getattr(self, f"find_{affix_type}")

            if affix_finder(substring):
                # Get the `TokenMeta` object of the affix along with updated
                # substring and start pos pointer after removing the affix
                token_meta, substring, pos = getattr(self, f"_get_{affix_type}_token_meta")(
                    substring, pos
                )

                affixes[f"{affix_type}"].append(token_meta)

                last_i = i

            # Change the affix type.
            i += 1

        # Get infix TokenMeta objects if any.
        if self.infix_matches(substring):
            infixes, substring, pos = self._get_infix_token_metas(substring, pos)
            affixes["infix"].extend(infixes)

        return substring, pos, affixes, exception_tokens

    def _attach_tokens(
        self,
        doc: Doc,
        substring: str,
        start_pos: int,
        space_after: bool,
        affixes: DefaultDict,
        exception_tokens: List[TokenMeta],
    ) -> Doc:
        """Attach all the `TokenMeta` objects which are the result of splitting affixes
        in Doc object's container. Returns Doc object.
       
        Args:
            doc: Original Document
            substring: The substring remaining after splitting all the affixes.
            start_pos: The pointer to location of start of substring in text.
            space_after: If there is a space after the original substring before splitting any affixes 
                in the text.
            affixes: Dict holding TokenMeta lists of each affix types(prefix, suffix, infix) 
                formed as the result of splitting affixes.
            exception_tokens: The list of TokenMeta object of exception tokens.

        Returns:
            doc: Document with all the TokenMeta objects of every token after splitting 
                affixes and exceptions.
        """

        # Append the prefix TokenMeta list in doc
        doc.container.extend(affixes["prefix"])

        # Append the exceptions TokenMeta list in doc
        doc.container.extend(exception_tokens)

        # If subtring is remaining after splitting all the affixes.
        if substring:
            # Create the TokenMeta object
            end_pos = start_pos + len(substring) - 1
            token_meta = TokenMeta(
                hash_key=self.vocab.store[(substring)],
                start_pos=start_pos,
                end_pos=end_pos,
                space_after=False,  # for the last token space_after will be updated explicitly according to the original substring.
                is_space=False,
            )

            # Append the token to the document
            doc.container.append(token_meta)

        # Append the infixes TokenMeta list in doc
        doc.container.extend(affixes["infix"])

        # Append the suffixes TokenMeta list in doc
        doc.container.extend(reversed(affixes["suffix"]))

        # Get the last token and update it's space_after attr according to original substring's TokenMeta data
        doc.container[-1].space_after = space_after

        return doc

    def _get_prefix_token_meta(self, substring: str, pos: int) -> Tuple[TokenMeta, str, int]:
        """Makes TokenMeta data for substring which are prefixes.

        Args:
            substring: The substring to tokenize.
            pos: The pointer to the start position of substring in the text.

        Returns:
            token_meta: The TokenMeta object with TokenMeta data of prefix.
            substring: The updated substring after removing prefix.
            pos: The pointer to the start position of new substring in the text.
        """

        # Get the length of prefix match in the substring.
        pre_len = self.find_prefix(substring)

        # break if pattern matches the empty string
        if pre_len == 0:
            return None, substring, pos

        end_pos = pos + pre_len - 1

        # Create the TokenMeta object
        token_meta = TokenMeta(
            hash_key=self.vocab.store[str(substring[:pre_len])],
            start_pos=pos,
            end_pos=end_pos,
            space_after=False,  # for the last token space_after will be updated explicitly according to the original substring.
            is_space=False,
        )

        pos = end_pos + 1

        # Update the remaining substring after removing the prefix.
        substring = substring[pre_len:]

        return token_meta, substring, pos

    def _get_suffix_token_meta(self, substring: str, pos: int) -> Tuple[TokenMeta, str, int]:
        """Makes TokenMeta data for substring suffixes.

        Args:
            substring: The `substring` to tokenize.
            pos: The pointer to the start position of substring in the text.

        Returns:
            token_meta: The TokenMeta object of the suffix.
            substring: The updated substring after removing the suffix.
            pos: The pointer to the start position of new `substring` in the text.
        """

        # Get the length of suffix match in the substring.
        suff_len = self.find_suffix(substring)

        # break if pattern matches the empty string
        if suff_len == 0:
            return None, substring

        # A pointer to the start of the suffix in the substring relative to the original text.
        pos_suffix = pos + len(substring) - suff_len

        # A pointer to the end of the suffix in the substring relative to the original text.
        end_pos_suffix = pos_suffix + suff_len - 1

        # Create the TokenMeta object
        token_meta = TokenMeta(
            hash_key=self.vocab.store[str(substring[len(substring) - suff_len :])],
            start_pos=pos_suffix,
            end_pos=end_pos_suffix,
            space_after=False,  # for the last token space_after will be updated explicitly in end.
            is_space=False,
        )

        # Update the remaining substring after removing the suffix.
        substring = substring[:-suff_len]

        return token_meta, substring, pos

    def _get_infix_token_metas(self, substring: str, pos: int) -> Tuple[List[TokenMeta], str, int]:
        """Makes list of TokenMeta data for substring which are infixes.

        Args:
            substring: The substring to tokenize.
            pos: The pointer to location of start of substring in text.

        Returns:
            infix_tokens_metas: the list of TokenMeta objects of infixes
                found in `substring`.
            substring: The updated substring after processing for all infixes.
            pos: The pointer to the start position of new `substring` in text.
        """

        # Get all the infix matches in list
        infixes = self.infix_matches(substring)

        # List to hold `TokenMeta` object of all infixes
        infix_tokens_metas = []

        # List holding start and end position of tokens
        # created due to infixes relative to substring, starting at 0
        positions = [0]

        for match in infixes:
            positions.extend([match.start(), match.end()])

        # Adding the endpos of last token
        positions.append(len(substring))

        for i in range(len(positions) - 1):
            # start and end postion of token relative to substring
            start_pos = positions[i]
            end_pos = positions[i + 1]

            # Check if at the end of the substring
            if start_pos == end_pos:
                break

            else:
                # Create the TokenMeta object
                # pos added to make `start_pos` and `end_pos` relative to orginal text.
                token_meta = TokenMeta(
                    hash_key=self.vocab.store[str(substring[start_pos:end_pos])],
                    start_pos=start_pos + pos,
                    end_pos=end_pos + pos,
                    space_after=False,  #  For this token space_after will be updated explicitly in end.
                    is_space=False,
                )

                # Append the token to the infix_list
                infix_tokens_metas.append(token_meta)

        # We have already proccesed the full substring so updating
        # pos just to have similar structure to other _get_affix_meta.
        pos = len(substring) + 1

        # There is no remaining substring
        substring = ""

        return infix_tokens_metas, substring, pos

    def _get_exception_token_metas(self, substring: str, pos: int) -> Tuple[List[TokenMeta], str]:
        """Make a list of TokenMeta objects of exceptions found in `substring`.

        Args:
            substring: The substring to tokenize.
            pos: The pointer to location of start of substring in text.

        Returns:
            exception_token_metas : the list of exceptions TokenMeta 
                objects.
            substring: The updated substring after processing the exceptions.

        """

        # List to hold TokenMeta objects of exceptions found in the `substring`.
        exception_token_metas = []

        for e in self.exceptions[substring]:
            ORTH = e["ORTH"]
            end_pos = pos + len(ORTH) - 1

            # Create the TokenMeta object
            token_meta = TokenMeta(
                hash_key=self.vocab.store[ORTH],
                start_pos=pos,
                end_pos=end_pos,
                space_after=False,  # for the last token space_after will be updated explicitly in end.
                is_space=False,
            )

            # Append the token to the  exception tokens list
            exception_token_metas.append(token_meta)

            # update start_pos for next orth
            pos = end_pos + 1

        # There is no remaining substring.
        substring = ""

        return exception_token_metas, substring

    def infix_matches(self, substring: str) -> List[Match]:
        """Find internal split points of the string, such as hyphens.
        
        Args:
            substring : The string to segment.

        Returns:
            A list of `re.MatchObject` objects that have `.start()`
                and `.end()` methods, denoting the placement of internal 
                segment separators, e.g. hyphens.
        """

        # Return empty list if no infix matches are in substring.
        if self.infix_finditer is None:
            return []

        # Return a list of MatchObject instances over all non-overlapping
        # matches for the infixes in the substring.
        return list(self.infix_finditer(substring))

    def find_prefix(self, substring: str) -> int:
        """Find the length of a prefix that should be segmented from the
        string, or None if no prefix rules match.

        Args:
            substring: The string to segment.
            
        Returns:
            The length of the prefix if present, otherwise 0.
        """

        # Return 0 if no prefix match is found in substring.
        if self.prefix_search is None:
            return 0

        # The MatchObject with the end and start postion of the prefix in the substring.
        match = self.prefix_search(substring)

        # Return the length of the prefix match in the substring.
        return (match.end() - match.start()) if match is not None else 0

    def find_suffix(self, substring: str) -> int:
        """Find the length of a suffix that should be segmented from the
        string, or None if no suffix rules match.

        Args:
            substring: The string to segment.

        Returns:
            The length of the suffix if present, otherwise 0.
        """

        # Return 0 if no suffix match is found in substring.
        if self.suffix_search is None:
            return 0

        # The MatchObject with the end and start postion of the suffix in the substring.
        match = self.suffix_search(substring)

        # Return the length of the suffix match in the substring.
        return (match.end() - match.start()) if match is not None else 0

    @staticmethod
    def simplify(worker, tokenizer: "Tokenizer"):
        """This method is used to reduce a `Tokenizer` object into a list of simpler objects that can be
           serialized.
        """

        # Simplify attributes

        model_name = pickle.dumps(tokenizer.vocab.model_name)

        return model_name

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Create an object of type Tokenizer from the reduced representation in `simple_obj`.

        Args:
            worker (BaseWorker) : The worker on which the new Tokenizer object is to be created.
            simple_obj (tuple) : A tuple resulting from the serialized then deserialized returned tuple
                                from the `_simplify` static method above.

        Returns:
           tokenizer (Tokenizer) : a Tokenizer object
        """

        # Get the tuple elements
        model_name = simple_obj

        # Unpickle
        model_name = pickle.loads(model_name)

        # Create the tokenizer object
        tokenizer = Tokenizer(vocab=model_name)

        return tokenizer
