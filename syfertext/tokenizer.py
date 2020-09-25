from .doc import Doc
from .vocab import Vocab
from .state import State
from .pointers import StatePointer
from .underscore import Underscore

from . import LOCAL_WORKER

from .token_exception import TOKENIZER_EXCEPTIONS

from .punctuations import TOKENIZER_PREFIXES
from .punctuations import TOKENIZER_SUFFIXES
from .punctuations import TOKENIZER_INFIXES
from .utils import hash_string
from .utils import search_resource
from .utils import create_state_query
from .utils import compile_suffix_regex
from .utils import compile_infix_regex
from .utils import compile_prefix_regex
from .utils import msgpack_code_generator

import re

from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.base import BaseWorker
from syft.generic.string import String
import syft.serde.msgpack.serde as serde

import pickle
from collections import defaultdict

from typing import List
from typing import Union
from typing import Tuple
from typing import Match
from typing import DefaultDict
from typing import Dict
from typing import Set


class TokenMeta(object):
    """This class holds some meta data about a token from the text held by a Doc object.
    This allows to create a Token object when needed.
    """

    def __init__(self, hash_key: int, space_after: bool):
        """Initializes a TokenMeta object

        Args:
            hash_key(int): hash value of the string stored by the Token object
            space_after (bool): Whether the token is followed by a single white
                space (True) or not (False).
        """

        # stores the hash of the hash of the string
        self.orth = hash_key

        self.space_after = space_after

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()


class Tokenizer(AbstractSendable):
    def __init__(
        self,
        exceptions: Dict[str, List[dict]] = None,
        prefixes: List[str] = None,
        suffixes: List[str] = None,
        infixes: List[str] = None,
    ):
        """Initializes the `Tokenizer` object. Pass in empty lists for suffix, prefix and infix
        if you don't want any suffix, prefix and infix rules and empty dict for no exception rules.
        If None(default value), is passed, we use pre-configured rules.

        Args:
            exceptions: Exception cases for the tokenizer.
                Example: "e.g.", "I'ma". The exception dict should
                specifiy how these exceptions are tokenized, e.g.,
                exceptions = {"e.g." : [{"ORTH": "e.g."}],
                              "I'ma" : [{"ORTH": "I"}, {"ORTH": "'m"},
                                        {"ORTH": "a"}]
                             }
                Other properties than "ORTH" can also be specified.
                see `token_exception.py` for more examples.
            prefixes: A list of strings to separate as prefixes during
                tokenization.
                Example: ["@"]. So in "@username", "@" will be separated as
                a prefix.
            suffixes: A list of strings to separate as suffixes during
                tokenization.
                Example: ["!"]. So in "Oh!", "!" will be separated as
                a suffix.
            infixes: A list of strings to separate as infixes during
                tokenization.
                Example: ["-"]. So in  "Hell-o", "-" will be separated as
                an infix.
        """

        super(Tokenizer, self).__init__()

        # Set the tokenization rules
        self.load_rules(
            exceptions=exceptions, prefixes=prefixes, suffixes=suffixes, infixes=infixes
        )

    @property
    def pipeline_name(self) -> str:
        """A getter for the `_pipeline_name` property.

        Returns:
           The lower cased `_pipeline_name` property.
        """

        return self._pipeline_name.lower()

    @pipeline_name.setter
    def pipeline_name(self, name: str) -> None:
        """Set the pipeline name to which this object belongs.

        Args:
            name: The name of the pipeline.
        """

        # Convert the name of lower case
        if isinstance(name, str):
            name = name.lower()

        self._pipeline_name = name

    @property
    def name(self) -> str:
        """A getter for the `_name` property.

        Returns:
           The lower cased `_name` property.
        """

        return self._name.lower()

    @name.setter
    def name(self, name: str) -> None:
        """Set the component name.

        Args:
            name: The name of the component
        """

        # Convert the name of lower case
        if isinstance(name, str):
            name = name.lower()

        self._name = name

    @property
    def access(self) -> Set[str]:
        """Get the access rules for this component.

        Returns:
            The set of worker ids where this component's state
            could be sent.
            If the string '*' is included in the set,  then all workers are
            allowed to receive a copy of the state. If set to None, then
            only the worker where this component is saved will be allowed
            to get a copy of the state.
        """

        return self._access_rules

    @access.setter
    def access(self, rules: Set[str]) -> None:
        """Set the access rules of this object.

        Args:
            rules: The set of worker ids where this component's state
                could be sent.
                If the string '*' is included in the set,  then all workers are
                allowed to receive a copy of the state. If set to None, then
                only the worker where this component is saved will be allowed
                to get a copy of the state.
        """

        self._access_rules = rules

    def load_rules(
        self,
        exceptions: Dict[str, List[dict]] = None,
        prefixes: List[str] = None,
        suffixes: List[str] = None,
        infixes: List[str] = None,
    ):
        """Sets/Resets the tokenization rules.

        Args:
            exceptions: Exception cases for the tokenizer.
                Example: "e.g.", "I'ma". The exception dict should
                specifiy how these exceptions are tokenized, e.g.,
                exceptions = {"e.g." : [{"ORTH": "e.g."}],
                              "I'ma" : [{"ORTH": "I"}, {"ORTH": "'m"},
                                        {"ORTH": "a"}]
                             }
                Other properties than "ORTH" can also be specified.
                see `token_exception.py` for more examples.
            prefixes: A list of strings to separate as prefixes during
                tokenization.
                Example: ["@"]. So in "@username", "@" will be separated as
                a prefix.
            suffixes: A list of strings to separate as suffixes during
                tokenization.
                Example: ["!"]. So in "Oh!", "!" will be separated as
                a suffix.
            infixes: A list of strings to separate as infixes during
                tokenization.
                Example: ["-"]. So in  "Hell-o", "-" will be separated as
                an infix.

        Modifies:
            properties `exceptions`, `prefix_search`, `suffix_search`,
               `infix_finditer`, `prefixes`, `suffixes`, and `infixes`
               are created by this method.


        """

        # If affixes are set to None, they should take the default
        # values
        if prefixes is not None:
            self.prefixes = prefixes
        else:
            self.prefixes = TOKENIZER_PREFIXES

        if suffixes is not None:
            self.suffixes = suffixes
        else:
            self.suffixes = TOKENIZER_SUFFIXES

        if infixes is not None:
            self.infixes = infixes
        else:
            self.infixes = TOKENIZER_INFIXES

        self.prefix_search = compile_prefix_regex(self.prefixes).search if self.prefixes else None
        self.suffix_search = compile_suffix_regex(self.suffixes).search if self.suffixes else None
        self.infix_finditer = compile_infix_regex(self.infixes).finditer if self.infixes else None

        if exceptions is not None:
            self.exceptions = exceptions
        else:
            self.exceptions = TOKENIZER_EXCEPTIONS

    def load_state(self) -> None:
        """Search for the state of this object on PyGrid.

        Modifies:
            self.vocab: The `vocab` property is initialized with the model
                 name. Its 'load_state()` method is also called.
        """

        # Start by creating the vocab and loading its state
        self.vocab = Vocab()
        self.vocab.pipeline_name = self.pipeline_name
        self.vocab.name = "vocab"
        self.vocab.owner = self.owner
        self.vocab.load_state()

        # Create the query. This is the ID according to which the
        # State object is searched on PyGrid
        state_id = create_state_query(pipeline_name=self.pipeline_name, state_name=self.name)

        # Search for the state
        result = search_resource(query=state_id, local_worker=self.owner)

        # If no state is found, return
        if not result:
            return

        # If a state is found get either its pointer if it is remote
        # or the state itself if it is local
        elif isinstance(result, StatePointer):
            # Get a copy of the state using its pointer
            state = result.get_copy()

        elif isinstance(result, State):
            state = result

        # Detail the simple object contained in the state
        exceptions_simple, prefixes_simple, suffixes_simple, infixes_simple = state.simple_obj

        exceptions = serde._detail(self.owner, exceptions_simple)
        prefixes = serde._detail(self.owner, prefixes_simple)
        suffixes = serde._detail(self.owner, suffixes_simple)
        infixes = serde._detail(self.owner, infixes_simple)

        # Load the state
        self.load_rules(
            exceptions=exceptions, prefixes=prefixes, suffixes=suffixes, infixes=infixes
        )

    def dump_state(self) -> State:
        """Returns a State object that holds the current state of this object.

        Returns:
            A State object that holds a simplified version of this object's state.
        """

        # Simplify the state variables
        exceptions_simple = serde._simplify(self.owner, self.exceptions)
        prefixes_simple = serde._simplify(self.owner, self.prefixes)
        suffixes_simple = serde._simplify(self.owner, self.suffixes)
        infixes_simple = serde._simplify(self.owner, self.infixes)

        # Create the query. This is the ID according to which the
        # State object is searched for on across workers
        state_id = f"{self.pipeline_name}:{self.name}"

        # Create the State object
        state = State(
            simple_obj=(exceptions_simple, prefixes_simple, suffixes_simple, infixes_simple),
            id=state_id,
            access=self.access,
        )

        return state

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
        # By meta data I mean the hash value of the string stored by the Token object
        # in the original text, and if the token is followed by a white space.

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
                    space_after=is_current_space,
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
                    space_after=is_current_space,
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
        """Tokenize each substring formed after splitting affixes and processing
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

        # If there is trailing space after the substring in text.
        space_after = token_meta.space_after

        # Get the remaining substring,affixes containing list of
        # TokenMeta for each type affix and list of TokenMeta of
        # exceptions after splitting the affixes.
        substring, affixes, exception_tokens = self._split_affixes(substring=substring)

        # Attach all the `TokenMeta` objects formed as result of splitting
        # the affixes and exception cases in the doc container.
        doc = self._attach_tokens(
            doc=doc,
            substring=substring,
            space_after=space_after,
            affixes=affixes,
            exception_tokens=exception_tokens,
        )

        return doc

    def _split_affixes(self, substring: str) -> Tuple[str, DefaultDict, List[TokenMeta]]:
        """Process substring for tokenizing prefixes, infixes, suffixes and exceptions.

        Args:
            substring: The substring to tokenize.

        Returns:
            substring: The substring to tokenize.
            affixes: Dict holding TokenMeta lists of each affix
                types as a result of splitting affixes
            exception_tokens: The list of exception tokens TokenMeta objects.
        """

        infixes = []
        exception_tokens = []

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
                exception_tokens, substring = self._get_exception_token_metas(substring)

                break

            # Get affix type (prefix or suffix)
            affix_type = next_affix[i % 2]

            affix_finder = getattr(self, f"find_{affix_type}")

            if affix_finder(substring):
                # Get the `TokenMeta` object of the affix along with updated
                # substring after removing the affix
                token_meta, substring = getattr(self, f"_get_{affix_type}_token_meta")(substring)

                affixes[f"{affix_type}"].append(token_meta)

                last_i = i

            # Change the affix type.
            i += 1

        # Get infix TokenMeta objects if any.
        if self.infix_matches(substring):
            infixes, substring = self._get_infix_token_metas(substring)
            affixes["infix"].extend(infixes)

        return substring, affixes, exception_tokens

    def _attach_tokens(
        self,
        doc: Doc,
        substring: str,
        space_after: bool,
        affixes: DefaultDict,
        exception_tokens: List[TokenMeta],
    ) -> Doc:
        """Attach all the `TokenMeta` objects which are the result of splitting affixes
        in Doc object's container. Returns Doc object.

        Args:
            doc: Original Document
            substring: The substring remaining after splitting all the affixes.
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
            token_meta = TokenMeta(
                hash_key=self.vocab.store[(substring)],
                space_after=False,  # for the last token space_after will be updated explicitly according to the original substring.
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

    def _get_prefix_token_meta(self, substring: str) -> Tuple[TokenMeta, str]:
        """Makes TokenMeta data for substring which are prefixes.

        Args:
            substring: The substring to tokenize.

        Returns:
            token_meta: The TokenMeta object with TokenMeta data of prefix.
            substring: The updated substring after removing prefix.
        """

        # Get the length of prefix match in the substring.
        pre_len = self.find_prefix(substring)

        # break if pattern matches the empty string
        if pre_len == 0:
            return None, substring

        # Create the TokenMeta object
        token_meta = TokenMeta(
            hash_key=self.vocab.store[str(substring[:pre_len])],
            space_after=False,  # for the last token space_after will be updated explicitly according to the original substring.
        )

        # Update the remaining substring after removing the prefix.
        substring = substring[pre_len:]

        return token_meta, substring

    def _get_suffix_token_meta(self, substring: str) -> Tuple[TokenMeta, str]:
        """Makes TokenMeta data for substring suffixes.

        Args:
            substring: The `substring` to tokenize.

        Returns:
            token_meta: The TokenMeta object of the suffix.
            substring: The updated substring after removing the suffix.
        """

        # Get the length of suffix match in the substring.
        suff_len = self.find_suffix(substring)

        # break if pattern matches the empty string
        if suff_len == 0:
            return None, substring

        # Create the TokenMeta object
        token_meta = TokenMeta(
            hash_key=self.vocab.store[str(substring[len(substring) - suff_len :])],
            space_after=False,  # for the last token space_after will be updated explicitly in end.
        )

        # Update the remaining substring after removing the suffix.
        substring = substring[:-suff_len]

        return token_meta, substring

    def _get_infix_token_metas(self, substring: str) -> Tuple[List[TokenMeta], str]:
        """Makes list of TokenMeta data for substring which are infixes.

        Args:
            substring: The substring to tokenize.

        Returns:
            infix_tokens_metas: the list of TokenMeta objects of infixes
                found in `substring`.
            substring: The updated substring after processing for all infixes.
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
                token_meta = TokenMeta(
                    hash_key=self.vocab.store[str(substring[start_pos:end_pos])],
                    space_after=False,  #  For this token space_after will be updated explicitly in end.
                )

                # Append the token to the infix_list
                infix_tokens_metas.append(token_meta)

        # There is no remaining substring
        substring = ""

        return infix_tokens_metas, substring

    def _get_exception_token_metas(self, substring: str) -> Tuple[List[TokenMeta], str]:
        """Make a list of TokenMeta objects of exceptions found in `substring`.

        Args:
            substring: The substring to tokenize.

        Returns:
            exception_token_metas : the list of exceptions TokenMeta
                objects.
            substring: The updated substring after processing the exceptions.

        """

        # List to hold TokenMeta objects of exceptions found in the `substring`.
        exception_token_metas = []

        for e in self.exceptions[substring]:
            ORTH = e["ORTH"]

            # Create the TokenMeta object
            token_meta = TokenMeta(
                hash_key=self.vocab.store[ORTH],
                space_after=False,  # for the last token space_after will be updated explicitly in end.
            )

            # Append the token to the  exception tokens list
            exception_token_metas.append(token_meta)

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
        name_simple = serde._simplify(worker, tokenizer.name)
        pipeline_name_simple = serde._simplify(worker, tokenizer.pipeline_name)

        return (name_simple, pipeline_name_simple)

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
        name_simple, pipeline_name_simple = simple_obj

        # Detail
        name = serde._detail(worker, name_simple)
        pipeline_name = serde._detail(worker, pipeline_name_simple)

        # Create the tokenizer object
        tokenizer = Tokenizer()
        tokenizer.pipeline_name = pipeline_name
        tokenizer.name = name
        tokenizer.owner = worker

        return tokenizer

    @staticmethod
    def get_msgpack_code() -> Dict[str, int]:
        """This is the implementation of the `get_msgpack_code()`
        method required by PySyft's SyftSerializable class.
        It provides a code for msgpack if the type is not present in proto.json.

        The returned object should be similar to:
        {
            "code": int value,
            "forced_code": int value
        }

        Both keys are optional, the common and right way would be to add only the "code" key.

        Returns:
            dict: A dict with the "code" and/or "forced_code" keys.
        """

        # If a msgpack code is not already generated, then generate one
        # the code is hash of class name
        if not hasattr(Tokenizer, "proto_id"):
            Tokenizer.proto_id = msgpack_code_generator(Tokenizer.__qualname__)

        code_dict = dict(code=Tokenizer.proto_id)

        return code_dict
