from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import pickle

from typing import List
from typing import Union

class DocPointer(ObjectPointer):

    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        tags: List[str] = None,
        description: str = None,
    ):

        super(DocPointer, self).__init__(location = location,
                                         id_at_location = id_at_location,
                                         owner = owner,
                                         id = id,
                                         garbage_collect_data = garbage_collect_data,
                                         tags = tags,
                                         description = description)



        
    @staticmethod
    def _simplify(doc_pointer):
        """
           This method is used to reduce a `DocPointer` object into a list of simpler objects that can be
           serialized 
        """

        # Simplify the attributes
        location_id = pickle.dumps(doc_pointer.location.id)
        tags = [pickle.dumps(tag) for tag in doc_pointer.tags] if doc_pointer.tags else None
        description = pickle.dumps(doc_pointer.description)
        
        # Get the path to the current class
        type_path = DocPointer.__module__ + '/' + DocPointer.__name__        

        return (location_id,
                doc_pointer.id_at_location,
                doc_pointer.id,
                doc_pointer.garbage_collect_data,
                tags,
                description,
                type_path)
    

    @staticmethod
    def _detail(worker: BaseWorker,
                simple_obj
    ):
        """
           Create an object of type DocPointer from the reduced representation in `simple_obj`.

           Parameters
           ----------
           worker: BaseWorker
                   The worker on which the new DocPointer object is to be created.
           simple_obj: tuple
                       A tuple resulting from the serialized then deserialized returned tuple
                       from the `_simplify` static method above.

           Returns
           -------
           doc_pointer: DocPointer
                      a DocPointer object, pointing to a Doc object
        """

        # Get the typle elements
        location_id, id_at_location, id, garbage_collect_data, tags, description = simple_obj

        # Unpickle
        location_id = pickle.loads(location_id)
        tags = [pickle.loads(tag) for tag in tags] if tags else None
        description = pickle.loads(description)


        # Get the worker `location` on which lives the pointed-to Doc object
        location = worker.get_worker(id_or_worker = location_id)
        
        # Create a DocPointer object
        doc_pointer = DocPointer(location = location,
                                 id_at_location = id_at_location,
                                 owner = worker,
                                 id = id,
                                 garbage_collect_data = garbage_collect_data,
                                 tags = tags,
                                 description = description
        )

        return doc_pointer
