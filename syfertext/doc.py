from .token import Token
import syft
import torch
import numpy as np

hook = syft.TorchHook(torch)


from syft.generic.abstract.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List
from typing import Dict
from typing import Set
from typing import Union
from typing import Generator
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
        # of this doc resides. However if it is passed explicitly
        # in init , we assign this client_id
        if client_id is not None:
            self.client_id = client_id

        # This list is populated in the __call__ method of the Tokenizer object.
        # Its members are objects of the TokenMeta class defined in the tokenizer.py
        # file
        self.container = list()

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
        assert self.has_attribute(name), f"Document does not have the attribute {name}"

        delattr(self._, name)

    def get_attribute(self, name: str):
        """Returns value of custom attribute with the name `name` if it is present, else raises `AttributeError`.

        Args:
            name (str): name of the custom attribute.

        Returns:
            value (obj): value of the custom attribute with name `name`.
        """

        return getattr(self._, name)

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
            doc_vector: doc vector
        """
        return self.get_vector()

    @property
    def vector_norm(self) -> torch.FloatTensor:
        """Get the L2 norm of the document vector

        Returns:
            A torch tensor representing the L2 norm of the document vector
        """

        vector = torch.tensor(self.vector)

        norm = (vector ** 2).sum()
        norm = torch.sqrt(norm)

        return norm

    def similarity(self, other: "Doc") -> torch.Tensor:
        """Compute the cosine similarity between two Doc vectors.

        Args:
            other (Doc): The Doc to compare with.

        Returns:
            Tensor: A cosine similarity score. Higher is more similar.
        """

        # Make sure both vectors have non-zero norms
        assert (
            self.vector_norm.item() != 0.0 and other.vector_norm.item() != 0.0
        ), "One of the provided vectors has a zero norm!"

        # Compute similarity
        sim = torch.dot(torch.tensor(self.vector), torch.tensor(other.vector))
        sim /= self.vector_norm * other.vector_norm

        return sim

    def get_vector(self, excluded_tokens: Dict[str, Set[object]] = None):
        """Get document vector as an average of in-vocabulary token's vectors,
        excluding token according to the excluded_tokens dictionary.

        Args
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency, sets of values.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            doc_vector: Document vector ignoring excluded tokens.
        """

        # Get the valid tokens which are to be included
        valid_tokens = self._get_valid_tokens(excluded_tokens)

        vectors = None

        # Count the tokens that have vectors
        vector_count = 0

        for token in valid_tokens:

            if token.has_vector:
                # Increment the vector counter
                vector_count += 1

                # Accumulate token's vector by summing them
                vectors = token.vector if vectors is None else vectors + token.vector

        # If no tokens with vectors were found, just get the default vector(zeros)
        if vector_count == 0:
            doc_vector = self.vocab.vectors.default_vector
        else:
            # The Doc vector, which is the average of all vectors
            doc_vector = vectors / vector_count
        return doc_vector

    def get_token_vectors(self, excluded_tokens: Dict[str, Set[object]] = None) -> torch.tensor:
        """Get the Numpy array of all the vectors corresponding to the tokens in the `Doc`,
        excluding token according to the excluded_tokens dictionary.

        Args
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            token_vectors: The torch tensor of shape - (No.of tokens, size of vector)
                containing all the vectors.
        """

        # Get the valid tokens which are to be included
        valid_tokens = self._get_valid_tokens(excluded_tokens)

        # The list for holding all token vectors.
        token_vectors = []

        for token in valid_tokens:
            # Get the vector of the token
            token_vectors.append(token.vector)

        # Convert to Numpy array.
        # token_vectors = np.array(token_vectors)
        # Convert to torch tensor.
        token_vectors = torch.stack(tensors=token_vectors)

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

    def _get_valid_tokens(
        self, excluded_tokens: Dict[str, Set[object]] = None
    ) -> Generator[Token, None, None]:
        """Handy function to handle the logic of excluding tokens while performing operations on Doc.

        Args:
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}
        Yields:
            A generator with valid tokens, i.e. tokens which are `not` to be excluded.
        """

        if excluded_tokens:
            # Enforcing that the values of the excluded_tokens dict are sets, not lists.
            excluded_tokens = {
                attribute: set(excluded_tokens[attribute]) for attribute in excluded_tokens
            }

            # Iterate over all tokens in doc
            for token in self:

                # Check if token can be included by comparing its attribute values
                # to those in excluded_tokens dictionary.
                include_token = all(
                    [
                        token.get_attribute(key) not in excluded_tokens[key]
                        for key in excluded_tokens.keys()
                        if token.has_attribute(key)
                    ]
                )

                if include_token:
                    yield token
        else:
            # All tokens are included
            for token in self:
                yield token

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
