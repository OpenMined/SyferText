from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.string_pointer import StringPointer
from syft.workers.base import BaseWorker
from syft.generic.string import String


from typing import List
from typing import Union

class TokenizerPointer(ObjectPointer):

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

        super(TokenizerPointer, self).__init__(location = location,
                                               id_at_location = id_at_location,
                                               owner = owner,
                                               id = id,
                                               garbage_collect_data = garbage_collect_data,
                                               tags = tags,
                                               description = description)
    


    def __call__(self,
                 text: StringPointer
    ):

        # For the moment, and to protect privacy this method accepts only StringPointer objects whos `location`
        # is the same location as `self.location`
        assert text.location == self.location, "StringPointer and TokenizerPointers do not belong to the same worker"

        # Get the id of the remote String object pointed to by text.
        text_id_at_location = text.id_at_location

        # Create the command
        args = []
        kwargs = {'text_id':text_id_at_location}
        
        command = ('__call__', self.id_at_location, args, kwargs)
        
        # Send the command
        response = self.owner.send_command(self.location, command)

        return response
