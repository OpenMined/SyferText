from .doc import Doc
from .vocab import Vocab
from .underscore import Underscore

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.string import String

import pickle
from typing import List, Union


class TokenMeta(object):
    """This class holds some meta data about a token from the text held by a Doc object.
       This allows to create a Token object when needed.
    """

    def __init__(self, start_pos: int, end_pos: int, space_after: bool, is_space: bool):
        """Initializes a TokenMeta object

        Args:
            start_pos (int): The start index of the token in the Doc text.
            end_pos (int): The end index of the token in the Doc text (the end index is
                part of the token).
            space_after (bool): Whether the token is followed by a single white 
                space (True) or not (False).
            is_space (bool): Whether the token itself is composed of only white 
                spaces (True) or not (false).

        """

        self.start_pos = start_pos
        self.end_pos = end_pos
        self.space_after = space_after
        self.is_space = is_space

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()


class Tokenizer:
    def __init__(
        self, vocab: Union[Vocab, str],
    ):
        """Initialize the Tokenizer object
           
        Args:
            vocab (str or Vocab) :If str, this should be the name of the 
                language model to build the Vocab object from. such as 
                'en_core_web_lg'. This is useful when the Tokenizer 
                object is sent to a remote worker. So it can rebuild
                its Vocab object from scratch instead of sending the Vocab object to
                the remote worker which might take too much network traffic.
    
        """

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

    def __call__(self, text: Union[String, str] = None, text_id: int = None):
        """The real tokenization procedure takes place here.

           As in the spaCy library. This is not exactly equivalent to 
           text.split(' '). Because tokens can be whitle spaces if two or
           more consecutive white spaces are found.

           Exampele:
              'I love apples' gives three tokens: 'I', 'love', 'apples'
              'I  love apples ' gives four tokens: 'I', ' ', 'love', 'apples'
              ' I love ' gives three tokens: ' ', 'I', 'love' (yes a single white space
              at the beginning is considered a token)

           Tokenizing this ways helps reconstructing the original string
           without loss of white spaces.
           I think that reconstructing the original string might be a good way
           to blindly verify the sanity of the blind tokenization process.


           Parameters
           ----------
           text: Syft String or str
                 The text to be tokenized
           text_id: int
                    the text id to be tokenized. The id can be used to get the object
                    from the worker registery
        """

        # Either the `text` or the `text_id` should be specified, they cannot be both None
        assert (
            text is not None or text_id is not None
        ), "`text` and `text_id` cannot be both None"

        # Create a document that will hold meta data of tokens
        # By meta data I mean the start and end positions of each token
        # in the original text, if the token is followed by a white space,
        # if the token itself is composed of white spaces or not, etc ...

        # If the text is not specified, then get the text using its id
        if text is None:
            text = self.owner.get_obj(text_id)

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

            # We are looking for a character that is the opposit of 'is_space'
            # if 'is_space' is True, then we want to find a character that is
            # not a space. and vice versa. This event marks the end of a token.
            is_current_space = char.isspace()
            if is_current_space != is_space:

                # Create the TokenMeta object that can be later used to retrieve the token
                # from the text
                token_meta = TokenMeta(
                    start_pos=pos,
                    end_pos=i - 1,
                    space_after=is_current_space,
                    is_space=is_space,
                )

                # Append the token to the document
                doc.container.append(token_meta)

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
                    start_pos=pos,
                    end_pos=None,  # text[pos:None] ~ text[pos:]
                    space_after=is_current_space,
                    is_space=is_space,
                )

                # Append the token to the document
                doc.container.append(token_meta)

        return doc

    @staticmethod
    def create_pointer(
        tokenizer,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
    ):
        """
           Creates a TokenizerPointer object that points to a Tokenizer object
           living in the worker 'location'.

           Returns:
                  a TokenizerPointer object
        """

        # I put the import here in order to avoid circular imports
        from .pointers.tokenizer_pointer import TokenizerPointer

        if id_at_location is None:
            id_at_location = tokenizer.id

        if owner is None:
            owner = tokenizer.owner

        tokenizer_pointer = TokenizerPointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=ptr_id,
            garbage_collect_data=garbage_collect_data,
        )

        return tokenizer_pointer

    @staticmethod
    def simplify(worker, tokenizer: "Tokenizer"):
        """
           This method is used to reduce a `Tokenizer` object into a list of simpler objects that can be
           serialized.
        """

        # Simplify attributes
        model_name = pickle.dumps(tokenizer.vocab.model_name)

        return model_name

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """
           Create an object of type Tokenizer from the reduced representation in `simple_obj`.

           Parameters
           ----------
           worker: BaseWorker
                   The worker on which the new Tokenizer object is to be created.
           simple_obj: tuple
                       A tuple resulting from the serialized then deserialized returned tuple
                       from the `_simplify` static method above.

           Returns
           -------
           tokenizer: Tokenizer
                      a Tokenizer object
        """

        # Get the tuple elements
        model_name = simple_obj

        # Unpickle
        model_name = pickle.loads(model_name)

        # Create the tokenizer object
        tokenizer = Tokenizer(vocab=model_name,)

        return tokenizer
