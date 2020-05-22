from .token import Token
import syft
import torch
import numpy as np

hook = syft.TorchHook(torch)


from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List
from typing import Dict
from typing import Set
from typing import Union
from .underscore import Underscore
from .span import Span
from .pointers.span_pointer import SpanPointer
from .utils import normalize_slice


class Doc(AbstractObject):
    def __init__(
        self,
        vocab,
        id: int = None,
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
        client_id: str = None,
    ):

        super(Doc, self).__init__(id=id, owner=owner, tags=tags, description=description)

        self.vocab = vocab

        # we assign the client_id in the __call__ method of the SubPipeline
        # This is used to keep track of the worker where the pointer
        # of this doc resides. However if it is passed explicitely
        # in init , we assign this client_id
        if client_id is not None:
            self.client_id = client_id

        # This list is populated in the __call__ method of the Tokenizer object.
        # Its members are objects of the TokenMeta class defined in the tokenizer.py
        # file
        self.container = list()

        # We use this to provide sentences
        # to the client at remote location using
        # `_get_a_sentence` method.
        self.sent_i = 0

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()

    def set_attribute(self, name: str, value: object):
        """Creates a custom attribute with the name `name` and
           value `value` in the Underscore object `self._`

        Args:
            name (str): name of the custom attribute.
            value (object): value of the custom named attribute.
        """

        # make sure that the name is not empty and does not contain any spaces
        assert (
            isinstance(name, str) and len(name) > 0 and (" " not in name)
        ), "Argument name should be a non-empty str type containing no spaces"

        setattr(self._, name, value)

    def has_attribute(self, name: str) -> bool:
        """Returns `True` if the Underscore object `self._` has an attribute `name`. otherwise returns `False` 

        Args:
            name (str): name of the custom attribute.
        
        Returns:
            attr_exists (bool): `True` if `self._.name` exists, otherwise `False`  
        """

        # `True` if `self._` has attribute `name`, `False` otherwise
        attr_exists = hasattr(self._, name)

        return attr_exists

    def remove_attribute(self, name: str):
        """Removes the attribute `name` from the Underscore object `self._`

        Args:
            name (str): name of the custom attribute.
        """

        # Before removing the attribute, check if it exists
        assert self.has_attribute(name), "Document does not have the attribute {}".format(name)

        delattr(self._, name)

    def __getitem__(self, key: Union[int, slice]) -> Union[Token, Span, int]:
        """Returns a Token object at position `key` or Span object using slice.


        Args:
            key (int or slice): The index of the token within the Doc, 
                or the slice of the Doc to return as a Span object.
        Returns:
            Token or Span or id of the Span object.
        """
        if isinstance(key, int):
            idx = 0
            if key < 0:
                idx = len(self) + key
            else:
                idx = key

            # Get the corresponding TokenMeta object
            token_meta = self.container[idx]

            # Create a Token object with owner same as the doc object
            token = Token(doc=self, token_meta=token_meta, position=key, owner=self.owner)

            return token

        if isinstance(key, slice):

            # Normalize slice to handle negative slicing
            start, end = normalize_slice(len(self), key.start, key.stop, key.step)

            # Create a new span object
            span = Span(self, start, end, owner=self.owner)

            # If the following condition is satisfied, this means that this
            # Doc is on a different worker (the Doc owner) than the one where
            # the Language object that operates the pipeline is located (the Doc client).
            # In this case we will create the Span at the same worker as
            # this Doc, and return its ID to the client worker where a SpanPointer
            # will be made out of  this id.
            if self.owner.id != self.client_id:

                # Register the Span on it's owners object store
                self.owner.register_obj(obj=span)

                # Return span_id using which we can create the SpanPointer
                return span.id

            return span

    def __len__(self):
        """Return the number of tokens in the Doc."""
        return len(self.container)

    def __iter__(self):
        """Allows to loop over tokens in `self.container`"""
        for i in range(len(self.container)):

            # Yield a Token object
            yield self[i]

    @property
    def text(self):
        """Returns the text present in the doc with whitespaces"""
        return "".join(token.text_with_ws for token in self)

    @property
    def vector(self):
        """Get document vector as an average of in-vocabulary token's vectors

        Returns:
          doc_vector: document vector
        """

        # Accumulate the vectors here
        vectors = None

        # Count the tokens that have vectors
        vector_count = 0

        for token in self:

            # Get the vector of the token if one exists
            if token.has_vector:

                # Increment the vector counter
                vector_count += 1

                # Cumulate token's vector by summing them
                vectors = token.vector if vectors is None else vectors + token.vector

        # If no tokens with vectors were found, just get the default vector(zeros)
        if vector_count == 0:
            doc_vector = self.vocab.vectors.default_vector
        else:
            # Create the final Doc vector
            doc_vector = vectors / vector_count

        return doc_vector

    @property
    def is_sentenced(self):
        """Check if the document has sentence boundaries assigned. This is
        defined as having at least one token other than the first where sent_start is not None.
        """

        # check for empty doc
        if len(self) == 0:
            return False

        return (self.container[0].is_sent_start != None) and (self.container[0].is_sent_start)

    @property
    def sents(self):
        """Iterate over the sentences in the document. Yields sentence `Span`
        objects.

        Returns:
            List of Spans: Sentences in the document.
        """
        assert (
            self.is_sentenced
        ), "Doc has not been sentencised yet, please use the sentenciser pipeline to divide doc into senetences"

        start = 0
        for i in range(1, len(self)):
            if self.container[i].is_sent_start:
                span = Span(self, start, i, owner=self.owner)
                span.client_id = self.client_id
                start = i
                yield span

        if start != len(self):
            span = Span(self, start, len(self), owner=self.owner)
            span.client_id = self.client_id
            yield span

    def _get_a_sentence(self):
        """Get a single sentence span with the starting
        index being `self.sent_i`
        """
        assert (
            self.is_sentenced
        ), "Doc has not been sentencised yet, please use the sentenciser pipeline to divide doc into senetences"

        span = None
        for i in range(self.sent_i + 1, len(self)):
            if self.container[i].is_sent_start:

                # create a span of the sentence
                span = Span(self, self.sent_i, i, owner=self.owner)
                span.client_id = self.client_id

                # update sent_i to the current index
                # as this is the start of a new sentence
                # when we next time call `_get_a_sentence`
                self.sent_i = i

                # we only provide a single sentence, so we break out
                break

        # We arrived at the end of the doc
        if span == None:
            span = Span(self, self.sent_i, len(self), owner=self.owner)
            span.client_id = self.client_id

            # reset the sent_i as we reached the end of the doc
            self.sent_i = 0

        if span.owner.id != span.client_id:

            self.owner.register_obj(span)

            return span.id

        return span

    def _num_sents(self):  # TODO: think of a faster way, maybe store len of sentences in doc itself
        """returns no. of sentences a doc has, used by the DocPointer primarily"""
        return len([sent for sent in self.sents])

    def get_vector(self, excluded_tokens: Dict[str, Set[object]] = None):
        """Get document vector as an average of in-vocabulary token's vectors,
        excluding token according to the excluded_tokens dictionary.

        Args
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency, sets of values.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            doc_vector: document vector ignoring excluded tokens
        """

        # if the excluded_token dict in None all token are included
        if excluded_tokens is None:
            return self.vector

        # enforcing that the values of the excluded_tokens dict are sets, not lists.
        excluded_tokens = {
            attribute: set(excluded_tokens[attribute]) for attribute in excluded_tokens
        }

        vectors = None

        # Count the tokens that have vectors
        vector_count = 0

        for token in self:

            # Get the vector of the token if one exists and if token is not excluded
            include_token = True

            include_token = all(
                [
                    getattr(token._, key) not in excluded_tokens[key]
                    for key in excluded_tokens.keys()
                    if hasattr(token._, key)
                ]
            )

            if token.has_vector and include_token:
                # Increment the vector counter
                vector_count += 1

                # Cumulate token's vector by summing them
                vectors = token.vector if vectors is None else vectors + token.vector

        # If no tokens with vectors were found, just get the default vector(zeros)
        if vector_count == 0:
            doc_vector = self.vocab.vectors.default_vector
        else:
            # Create the final Doc vector
            doc_vector = vectors / vector_count
        return doc_vector

    def get_token_vectors(self, excluded_tokens: Dict[str, Set[object]] = None) -> np.ndarray:
        """Get the Numpy array of all the vectors corresponding to the tokens in the `Doc`,
        excluding token according to the excluded_tokens dictionary.

        Args
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            token_vectors: The Numpy array of shape - (No.of tokens, size of vector) 
                containing all the vectors.
        """

        # enforcing that the values of the excluded_tokens dict are sets, not lists.
        if excluded_tokens is not None:
            excluded_tokens = {
                attribute: set(excluded_tokens[attribute]) for attribute in excluded_tokens
            }

        # The list for holding all token vectors.
        token_vectors = []

        for token in self:

            # Get the vector of the token if the token is not excluded
            include_token = True

            if excluded_tokens is not None:
                include_token = all(
                    [
                        getattr(token._, key) not in excluded_tokens[key]
                        for key in excluded_tokens.keys()
                        if hasattr(token._, key)
                    ]
                )

            if include_token:
                token_vectors.append(token.vector)

        # Convert to Numpy array.
        token_vectors = np.array(token_vectors)

        return token_vectors

    def get_encrypted_vector(
        self,
        *workers: BaseWorker,
        crypto_provider: BaseWorker = None,
        requires_grad: bool = True,
        excluded_tokens: Dict[str, Set[object]] = None,
    ):
        """Get the mean of the vectors of each Token in this documents.

        Args:
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency, sets of values.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            Tensor: A tensor representing the SMPC-encrypted vector of this document.
        """

        # You need at least two workers in order to encrypt the vector with SMPC
        assert len(workers) > 1

        # Storing the average of vectors of each in-vocabulary token's vectors
        doc_vector = self.get_vector(excluded_tokens=excluded_tokens)

        # Create a Syft/Torch tensor
        doc_vector = torch.Tensor(doc_vector)

        # Encrypt the vector using SMPC with PySyft
        doc_vector = doc_vector.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return doc_vector

    def get_encrypted_token_vectors(
        self,
        *workers: BaseWorker,
        crypto_provider: BaseWorker = None,
        requires_grad: bool = True,
        excluded_tokens: Dict[str, Set[object]] = None,
    ) -> torch.Tensor:
        """Get the Numpy array of all the vectors corresponding to the tokens in the `Doc`,
        excluding token according to the excluded_tokens dictionary.


        Args:
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography 
                (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency, 
                sets of values.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            Tensor: A SMPC-encrypted tensor representing the array of all vectors in this document, 
                ingonoring the excluded token.
        """

        "You need at least two workers in order to encrypt the vector with SMPC"
        assert len(workers) > 1

        # The array of all vectors corresponding to the tokens in `Doc`.
        token_vectors = self.get_token_vectors(excluded_tokens=excluded_tokens)

        # Create a Syft/Torch tensor
        token_vectors = torch.Tensor(token_vectors)

        # Encrypt the tensor using SMPC with PySyft
        token_vectors = token_vectors.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return token_vectors

    @staticmethod
    def create_pointer(
        doc,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
    ):
        """Creates a DocPointer object that points to a Doc object living in the the worker 'location'.

        Returns:
            DocPointer: pointer object to a document
        """

        # I put the import here in order to avoid circular imports
        from .pointers.doc_pointer import DocPointer

        if id_at_location is None:
            id_at_location = doc.id

        if owner is None:
            owner = doc.owner

        doc_pointer = DocPointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=ptr_id,
            garbage_collect_data=garbage_collect_data,
        )

        return doc_pointer
