from ..doc import Doc
from ..pointers.doc_pointer import DocPointer

import syft as sy
from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.string import String
from syft.generic.pointers.string_pointer import StringPointer
from syft.generic.pointers.object_pointer import ObjectPointer
import syft.serde.msgpack.serde as serde

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
            pipes: List[callable] = None
    ):
        """Initializes the object from a list of pipes.

        Initialization from a list of pipes is optional. This is
        only done by the detailer after the SubPipeline object 
        is sent to a remote worker. 
        When the SubPipeline is created on the local worker by
        the Language object, it is not assigned any list of pipe 
        components; the subpipeline is instead created from a 
        template that is loaded using the method `load_template`.

        Args:
            pipes (list of callables, optional): The list of pipe
                components.
        """

        # Set the id of the worker that owns the pipeline
        # that contains this subpipeline object.
        # The PySyft local worker is always the one
        # that instantiates subpipelines, so it is
        # the client of all subpipelines.
        self.client_id = self.owner.id

        # Create the subpipeline
        self.subpipeline = pipes
        
        super(SubPipeline, self).__init__(owner=self.owner)
        

    def load_template(self,
                      template: Dict[str, Union[bool, List[str]]],
                      factories: Dict[str, callable]
    ):
        """Loads the subpipeline template.


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

        # set the pipe names property
        self.pipe_names = template['names']
        
        # Create the subpipeline property
        self.subpipeline = [factories[name].factory() for name in template['names']]

    def send(
            self,
            location: BaseWorker
    ):
        """Sends this object to the worker specified by 'location'. 

        Args:
            location (BaseWorker): The BaseWorker object to which the object is 
                to be sent. Note that this is never actually the BaseWorker but instead
                a class which inherits the BaseWorker abstraction.

            Returns:
                (SubPipelinePointer): A pointer to this object.
        """

        ptr = self.owner.send(self, location)
        
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


    @staticmethod
    def simplify(
            worker: BaseWorker,
            subpipeline: "SubPipeline"
    ):
        """Simplifies a SubPipeline object. 

        This requires simplifying each underlying pipe
        component.

        Args:
            worker (BaseWorker): The worker on which the
                simplify operation is carried out.
                Required by PySyft serde, but unused here.
            subpipeline (SupPipeline): the SubPipeline object
                to simplify.

        Returns:
            (tuple): The simplified SubPipeline object.
        
        """

        # Simplify the attributes and pipe components
        client_id = sy.serde.msgpack.serde._simplify(worker, subpipeline.client_id)
        pipe_names = sy.serde.msgpack.serde._simplify(worker, subpipeline.pipe_names)
        
        # A list to store the simplified pipes
        simple_pipes = []

        # Simplify each pipe
        for pipe in subpipeline.subpipeline:
            simple_pipes.append(
                (pipe.proto_id, pipe.simplify(worker, pipe))
                )


        
        return (client_id, pipe_names, simple_pipes)


    def detail(
            worker: BaseWorker,
            simple_obj: tuple
    ):
        """Takes a simplified SubPipeline object, details it along with
        every pipe included in it and returns a SubPipeline object.

        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
                Required by PySyft serde, but unused here.
            simple_obj (tuple): the simplified SubPipeline object.
        Returns:
            (SubPipeline): The SubPipeline object.
        """

        
        # Unpack the simplified object
        client_id, simple_pipe_names, simple_pipes = simple_obj

        # Detail the client ID and the pipe names
        client_id = sy.serde.msgpack.serde._detail(worker, client_id)
        pipe_names = sy.serde.msgpack.serde._detail(worker, simple_pipe_names)        

        # Initialize a list of pipes
        pipes = []
        
        # Detail the pipes with the help of PySyft serde module
        for simple_pipe in simple_pipes:

            # Get the proto id of the pipe
            proto_id, simple_pipe = simple_pipe

            # Detail the simple_pipe to retriev the pipe object
            pipe = serde.detailers[proto_id](worker, simple_pipe)

            pipes.append(pipe)

        # Create the subpipeline object and set the client ID
        subpipeline = SubPipeline(pipes = pipes)
        subpipeline.client_id = client_id
        subpipeline.pipe_names = pipe_names


        return subpipeline
               

    def __repr__(self):

        # Create a list of pipe names included in the subpipeline
        pipe_names = "[" + " > ".join(self.pipe_names) + "]"
        
        return self.__class__.__name__ + pipe_names
