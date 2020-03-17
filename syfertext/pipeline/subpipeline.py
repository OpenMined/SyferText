from ..doc import Doc
from ..pointers.doc_pointer import DocPointer

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.string import String
from syft.generic.pointers.string_pointer import StringPointer
from syft.generic.pointers.object_pointer import ObjectPointer

import pickle
from typing import Union, Dict, List

class SubPipeline(AbstractObject):
    """This class defines a subpipeline. A subpipeline
    is an PySyft object that encapsulate one or more
    pipe components that operate on the same worker.

    At initialization of SyferText, a `owner` property 
    is assigned to this class, and holds the PySyft 
    local worker as the default owner.
    """
    
    def __init__(
            self,
            template: Dict[str, Union[bool, List[str]]],
            factories: Dict[str, callable]
    ):
        """Initializes the object from a template.

        Args:
            template (dict): This is a dictionary representing
                the subpipeline template. Here is an example of
                how a template looks like: 
                {'remote': True, 'names': ['tokenizer', 'tagger']}
            factories (dict): This is a dictionary that contains 
                a mapping between a pipe name and the object that
                knows how to create such a pipe using a factory
                method. Example to create a tokenizer:
                factories['tokenizer'].factory()
        """

        # Set the id of the worker that owns the pipeline
        # that contains this subpipeline object.
        # The PySyft local worker is always the one
        # that instantiates subpipelines, so it is
        # the client of all subpipelines.
        self.client_id = self.owner.id
        
        # Create the subpipeline
        self.subpipeline = [factories[name].factory() for name in template['names']]

        
        super(SubPipeline, self).__init__(owner=self.owner)
        

    def __call__(self,
                 input: Union[str, String, Doc],
    ) -> Union[int, str, Doc]:
        """Execute the subpipeline.

        Args:
            input (str, String, Doc): The input on which the 
                subpipeline operates. It could be either the text 
                or it could be the Doc to modify.

        Return:
            (int, str, Doc): Either the modified Doc object,
                or the ID of that Doc object (str or int).
        """


        # Execute the first pipe in the subpipeline
        doc = self.subpipeline[0](input)

        
        # Execute the  rest of pipes in the subpipeline 
        for pipe in self.subpipeline:
            doc = pipe(doc)


        # If the Language object using this subpipeline
        # is located on a different worker, then
        # return the id of the Doc object, not the Doc
        # object itself. This id will be used by the
        # Language object to create a DocPointer object
        if self.client_id != self.owner.id:

            # Register the Doc in the current worker's
            # object store
            self.owner.register_obj(obj=doc)

            # Return the Doc's ID
            return doc.id

        # Otherwise, return the Doc object itself
        return doc

