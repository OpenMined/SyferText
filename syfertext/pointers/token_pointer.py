from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import syft as sy

from typing import List
from typing import Union


class TokenPointer(ObjectPointer):
    """An Object Pointer that points to the Token Object at remote location"""

    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
    ):

        super(TokenPointer, self).__init__(
            location=location, id_at_location=id_at_location, owner=owner, id=id,
        )

    @property
    def text_(self):
        """Get the token text in Syft's String type"""

        # Create the command
        command = ("text_", self.id_at_location, [], {})

        # Send the command
        text = self.owner.send_command(self.location, command)

        return text

    def __len__(self):
        """Get the length of the token."""

        # Create the command
        command = ("__len__", self.id_at_location, [], {})

        # Send the command
        length = self.owner.send_command(self.location, command)

        return length
