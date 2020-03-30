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
from typing import List, Union, Tuple, Match


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
        vocab: Union[Vocab, str],
        rules=TOKENIZER_EXCEPTIONS,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
    ):
        """Initialize the Tokenizer object
           
            Args:
                vocab: str or Vocab object
                        If str, this should be the name of the language model to build the 
                        Vocab object from. such as 'en_core_web_lg'. This is useful when
                        the Tokenizer object is sent to a remote worker. So it can rebuild
                        its Vocab object from scratch instead of send the Vocab object to
                        the remote worker which might take too much network traffic.
                rules (dict): Exceptions and special-cases for the tokenizer.
                prefix_search(callable): A function matching the signature of
                        `re.compile(string).search` to match prefixes.
                suffix_search(callable): A function matching the signature of
                        `re.compile(string).search` to match sufixes.
                infix_finditer(callable): A function matching the signature of
                        `re.compile(string).finditer` to match infixes.
        """
        self.prefix_search = prefix_search
        self.suffix_search = suffix_search
        self.infix_finditer = infix_finditer
        if rules:
            self.special_cases = rules
        else:
            self.special_cases = {}

        if isinstance(vocab, Vocab):
            self.vocab = vocab
        else:
            self.vocab = Vocab(model_name=vocab)

    def factory(self):
        """Creates a clone of this object.
        This method is used by the SupPipeline class to create
        objects using subpipeline templates.
        """

        return Tokenizer(vocab=self.vocab)

    def __call__(self, text: Union[String, str]):
        """The real tokenization procedure takes place here.
        As in the spaCy library. This is not exactly equivalent to 
        text.split(' '). Because tokens can be whitle spaces if two or
        more consecutive white spaces are found.And affixes are splited.
        Exampele:
            'I love apples' gives three tokens: 'I', 'love', 'apples'
            'I  love apples ' gives four tokens: 'I', ' ', 'love', 'apples'
            ' I love ' gives three tokens: ' ', 'I', 'love' (yes a single white space
            at the beginning is considered a token)
            'I love-apples' gives 4 tokens: 'I', 'love', '-', 'apples'(infix is 
            tokenized seprately)
        Tokenizing this ways helps reconstructing the original string
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
        doc = Doc(self.vocab, text)

        # The number of characters in the text
        text_size = len(text)

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
        """ Tokenize each substring formed after spliting affixes and processing 
            exceptions(special cases). Returns Doc object.

        Args:
            substring (str) : The substring to tokenize.
            token_meta (TokenMeta) : The TokenMeta object of original substring
                before spliting affixes and special cases.
            doc (Doc): Document object. 

        Returns:    
            Document with all the TokenMeta objects of every token after spliting 
            affixes and exceptions.
        """

        pos = token_meta.start_pos
        space_after = token_meta.space_after
        (substring, pos, prefixes, suffixes, infixes, special_tokens,) = self._split_affixes(
            substring, pos
        )
        doc = self._attach_tokens(
            doc, substring, pos, space_after, prefixes, suffixes, infixes, special_tokens,
        )

        return doc

    def _split_affixes(
        self, substring: str, start_pos: int
    ) -> Tuple[str, int, List[TokenMeta], List[TokenMeta], List[TokenMeta], List[TokenMeta]]:
        """Process substring for tokenizinf prefixes, infixes, suffixes and exceptions.

        Args:
            substring (str) : The substring to tokenize.
            start_pos (int) : The pointer to location of start of substring in text.

        Returns:    
            substring: The substring to tokenize.
            start_pos: The pointer to location of start of substring in text.
            prefixes: The list of prefixes TokenMeta objects.
            suffixes: The list of suffixes TokenMeta objects.
            infixes: The list of infixes TokenMeta objects.
            special_tokens: The list of special tokens TokenMeta objects.
        """
        suffixes = []
        prefixes = []
        infixes = []
        special_tokens = []
        pos = start_pos
        end_pos = pos

        if substring:
            is_space = False
            space_after = False  # for the last token meta space after will be updated explicitely according to the original substring.
            while self.find_prefix(substring) or self.find_suffix(substring):
                if substring in self.special_cases:
                    break
                if self.find_prefix(substring):
                    token_meta, substring, pos = self._get_prefix_token_meta(substring, pos)
                    if token_meta:
                        # Append the suffix tokenmeta to the suffixes list to be added in doc container
                        prefixes.append(token_meta)
                    if substring in self.special_cases:
                        break
                if self.find_suffix(substring):
                    token_meta, substring = self._get_suffix_token_meta(substring, pos)
                    if token_meta:
                        # Append the suffix token meta to the suffixes list to be added in doc container
                        suffixes.append(token_meta)

            if substring in self.special_cases:
                # special cases token meta list to be added in doc container
                special_tokens = self._get_special_cases_token_meta(substring, pos)
                substring = ""

            elif self.find_infix(substring):
                infixes = self._get_infix_token_meta(substring, pos)
                substring = ""

        return substring, pos, prefixes, suffixes, infixes, special_tokens

    def _attach_tokens(
        self,
        doc: Doc,
        substring: str,
        start_pos: int,
        space_after: bool,
        prefixes: List[TokenMeta],
        suffixes: List[TokenMeta],
        infixes: List[TokenMeta],
        special_tokens: List[TokenMeta],
    ) -> Doc:
        """Attach all the TokenMeta objects in Doc object's container. Returns Doc object.

        Args:
            doc (Doc) : Original Document
            substring (str): The substring to tokenize.
            start_pos (int): The pointer to location of start of substring in text.
            space_after : TokenMeta object attribute from the substring befor sliting affixes
            prefixes (List[TokenMeta])  : The list of prefixes TokenMeta objects.
            suffixes (List[TokenMeta])  : The list of suffixes TokenMeta objects.
            infixes (List[TokenMeta])  : The list of infixes TokenMeta objects.
            special_tokens (List[TokenMeta])  : The list of special tokens TokenMeta objects.

        Returns:
            Document with all the TokenMeta objects of every token after spliting 
            affixes and exceptions.
        """

        if len(prefixes):
            doc.container.extend(prefixes)
        if len(special_tokens):
            doc.container.extend(special_tokens)
        if substring:
            # Create the TokenMeta object that can be later used to retrieve the token
            # from the text
            end_pos = start_pos + len(substring) - 1
            token_meta = TokenMeta(
                hash_key=self.vocab.store[(substring)],
                start_pos=start_pos,
                end_pos=end_pos,
                space_after=False,  # for the last token space after will be updated explicitely according to the original substring.
                is_space=False,
            )
            # Append the token to the document
            doc.container.append(token_meta)
            substring = ""
        if len(infixes):
            doc.container.extend(infixes)
        if len(suffixes):
            doc.container.extend(reversed(suffixes))

        # Get the last token and update it's space_after attr according to original substring's meta data
        last_token_meta = doc.container.pop()
        last_token_meta.space_after = space_after
        doc.container.append(last_token_meta)

        return doc

    def _get_prefix_token_meta(self, substring: str, pos: int) -> Tuple[TokenMeta, str, int]:
        """Makes token meta data for substring which are prefixes.

        Args:
            substring (str): The substring to tokenize.
            pos (int): The pointer to location of start of substring in text.

        Returns:
            token_meta: The TokenMeta object with token meta data of prefix.
            substring: The updated substring after removing prefix.
            pos: The pointer to location of start of updated substring in text.
        """
        pre_len = self.find_prefix(substring)
        # break if pattern matches the empty string
        if pre_len == 0:
            return None, substring, pos
        end_pos = pos + pre_len - 1
        # Create the TokenMeta object that can be later used to retrieve the token
        # from the text
        token_meta = TokenMeta(
            hash_key=self.vocab.store[str(substring[ : pre_len])],
            start_pos=pos,
            end_pos=end_pos,
            space_after=False,  # for the last token space after will be updated explicitely according to the original substring.
            is_space=False,
        )
        pos = end_pos + 1
        substring = substring[pre_len:]
        return token_meta, substring, pos

    def _get_suffix_token_meta(self, substring: str, pos: int) -> Tuple[TokenMeta, str]:
        """Makes token meta data for substring which are suffixes.

        Args:
            substring (str): The substring to tokenize.
            pos (int): The pointer to location of start of substring in text.

        Returns:
            token_meta: The TokenMeta object with token meta data of suffix.
            substring: The updated substring after removing suffix.
        """
        suff_len = self.find_suffix(substring)

        # break if pattern matches the empty string
        if suff_len == 0:
            return None, substring
        pos_suffix = pos + len(substring) - suff_len
        end_pos_suffix = pos_suffix + suff_len - 1
        # Create the TokenMeta object that can be later used to retrieve the token
        # from the text
        token_meta = TokenMeta(
            hash_key=self.vocab.store[str(substring[ len(substring)-suff_len: ])],
            start_pos=pos_suffix,
            end_pos=end_pos_suffix,
            space_after=False,  # for the last token space after will be updated explicitely in end.
            is_space=False,
        )
        substring = substring[:-suff_len]
        return token_meta, substring

    def _get_infix_token_meta(self, substring: str, pos: int) -> List[TokenMeta]:
        """Makes list of token meta data for substring which are infixes.

        Args:
            substring (str): The substring to tokenize.
            pos (int): The pointer to location of start of substring in text.

        Returns:
            infix_tokens_metas: the list of infixes TokenMeta 
            objects with tokens meta data of infixes.
        """
        infixes = self.find_infix(substring)
        offset = 0
        end_pos = 0
        infix_tokens_metas = []
        for match in infixes:
            if substring[offset : match.start()]:
                # Create the TokenMeta object that can be later used to retrieve the token
                # from the text
                end_pos = pos + len(substring[offset : match.start()]) - 1
                token_meta = TokenMeta(
                    hash_key=self.vocab.store[str(substring[offset : match.start()])],
                    start_pos=pos,
                    end_pos=end_pos,
                    space_after=False,  # for the last token space after will be updated explicitely in end.
                    is_space=False,
                )
                # Append the token to the infix_list
                infix_tokens_metas.append(token_meta)
                pos = end_pos + 1

            if substring[match.start() : match.end()]:
                # Create the TokenMeta object that can be later used to retrieve the token
                # from the text
                end_pos = pos + len(substring[match.start() : match.end()]) - 1
                token_meta = TokenMeta(
                    hash_key=self.vocab.store[str(substring[match.start() : match.end()])],
                    start_pos=pos,
                    end_pos=end_pos,
                    space_after=False,  # for the last token space after will be updated explicitely in end.
                    is_space=False,
                )
                # Append the token to the infix_list
                infix_tokens_metas.append(token_meta)
                pos = end_pos + 1
            offset = match.end()

        if substring[offset:]:
            # Create the TokenMeta object that can be later used to retrieve the token
            # from the text
            pos = end_pos + 1
            end_pos = pos + len(substring[offset:]) - 1
            token_meta = TokenMeta(
                hash_key=self.vocab.store[str(substring[offset : ])],
                start_pos=pos,
                end_pos=end_pos,
                space_after=False,  # for the last token space after will be updated explicitely in end.
                is_space=False,
            )
            # Append the token to the infix_list
            infix_tokens_metas.append(token_meta)
            pos = end_pos + 1

        return infix_tokens_metas

    def _get_special_cases_token_meta(self, substring: str, pos: int) -> List[TokenMeta]:
        """Makes list of token meta data for substring which are exceptions(special cases).

        Args:
            substring (str): The substring to tokenize.
            pos (int): The pointer to location of start of substring in text.

        Returns:
            special_cases_tokens_meta (List[TokenMeta]): the list of special cases TokenMeta 
            objects.

        """
        special_cases_tokens_meta = []
        for e in self.special_cases[substring]:
            ORTH = e["ORTH"]
            end_pos = pos + len(ORTH) - 1
            token_meta = TokenMeta(
                hash_key=self.vocab.store[ORTH],
                start_pos=pos,
                end_pos=end_pos,
                space_after=False,  # for the last token space after will be updated explicitely in end.
                is_space=False,
            )
            # Append the token to the special cases tokens list
            special_cases_tokens_meta.append(token_meta)
            # update start_pos for next orth
            pos = end_pos + 1
        return special_cases_tokens_meta

    def find_infix(self, str: str) -> List[Match]:
        """Find internal split points of the string, such as hyphens.
        
        Args:
            str (String): The string to segment.

        Returns:
            A list of `re.MatchObject` objects that have `.start()`
            and `.end()` methods, denoting the placement of internal segment
            separators, e.g. hyphens.
        """
        if self.infix_finditer is None:
            return 0
        return list(self.infix_finditer(str))

    def find_prefix(self, str: str) -> int:
        """Find the length of a prefix that should be segmented from the
        string, or None if no prefix rules match.

        Args:
            str (String): The string to segment.
            
        Returns:
            The length of the prefix if present, otherwise `None`.
        """
        if self.prefix_search is None:
            return 0
        match = self.prefix_search(str)
        return (match.end() - match.start()) if match is not None else 0

    def find_suffix(self, str: str) -> int:
        """Find the length of a suffix that should be segmented from the
        string, or None if no suffix rules match.

        Args:
            str (String): The string to segment.

        Returns:
            The length of the suffix if present, otherwise `None`.
        """
        if self.suffix_search is None:
            return 0
        match = self.suffix_search(str)
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
