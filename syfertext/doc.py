from .token import Token
import syft
import torch
hook = syft.TorchHook(torch)

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List
from typing import Union

class Doc(AbstractObject):

    def __init__(self,
                 vocab,
                 text,
                 id: int = None,                 
                 owner: BaseWorker = None,
                 tags: List[str] = None,
                 description: str = None
    ):

        super(Doc, self).__init__(id = id,
                                  owner = owner,
                                  tags = tags,
                                  description = description)        

        
        self.vocab = vocab
        self.text = text
        
        # This list is populated in the __call__ method of the Tokenizer object.
        # Its members are objects of the TokenMeta class defined in the tokenizer.py
        # file
        self.container = list()


    def __getitem__(self,
                    key: int
    ):
        """
           Returns a Token object at position `key`.

           Parameters
           ----------
           key: int
                the index of the token to return. 
                Example: 0 -> first token
                         1 -> second token
                         :
                         :
        """

        # Get the corresponding TokenMeta object
        token_meta =  self.container[key]

        # The start and stop positions of the token in self.text
        # notice that stop_position refers to one position after `token_meta.end_pos`.
        # this is practical for indexing
        start_pos = token_meta.start_pos
        stop_pos = token_meta.end_pos + 1 if token_meta.end_pos is not None else None
        
        # Create a Token object
        token = Token(doc = self,
                      #string = self.text[start_pos:end_pos],
                      start_pos = start_pos,
                      stop_pos = stop_pos,
                      is_space = token_meta.is_space,
                      space_after = token_meta.space_after)

        return token
                      


    @staticmethod
    def create_pointer(doc,
                       location: BaseWorker = None,
                       id_at_location: (str or int) = None,
                       register : bool = False,
                       owner: BaseWorker = None,
                       ptr_id: (str or int) = None,
                       garbage_collect_data: bool = True,
    ):
        """
           Creates a DocPointer object that points to a Doc object
           living in the the worker 'location'.

           Returns:
                  a DocPointer object
        """

        # I put the import here in order to avoid circular imports
        from .pointers.doc_pointer import DocPointer

        if id_at_location is None:
            id_at_location = doc.id

        if owner is None:
            owner = doc.owner
            
        doc_pointer =  DocPointer(location = location,
                                  id_at_location = id_at_location,
                                  owner = owner,
                                  id = ptr_id,
                                  garbage_collect_data = garbage_collect_data)

        return doc_pointer
    

    def __len__(self):
        """
           Return the number of tokens in the Doc.
        """

        return len(self.container)


    def __iter__(self):
        """
           Allows to loop over tokens in `self.container`
        """

        for i in range(len(self.container)):

            # Yield a Token object
            yield self[i]


    def getEncryptedVector(self, *workers):
        """
           Create one big vector composed of the concatenated Token vectors included in the
           Doc. The returned vector is SMPC-encrypted.

           TODO: This method should probably be removed. It served for a prototype test,
                 but concatenating all token vectors of the Doc into one big vector
                 might not be really useful for practical usecases.
        """
        
        assert len(workers) > 1, "You need at least two workers in order to encrypt the vector with SMPC"

        # Accumulate the vectors here
        vectors = []
        
        for token in self:
            
            # Get the encypted vector of the token
            vectors.append(token.getEncryptedVector(*workers))


        # Create the final Doc vector
        doc_vector = torch.cat(vectors)

        return doc_vector
