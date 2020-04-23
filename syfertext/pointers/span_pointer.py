from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import syft as sy

from typing import List
from typing import Union

from syfertext.pointers.token_pointer import TokenPointer


class SpanPointer(ObjectPointer):
    """An Object Pointer that points to the Span Object at remote location"""

    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
    ):

        super(SpanPointer, self).__init__(
            location=location, id_at_location=id_at_location, owner=owner, id=id,
        )

    def __len__(self):

        # Create the command
        command = ("__len__", self.id_at_location, [], {})

        # Send the command
        length = self.owner.send_command(self.location, command)

        return length

    def __getitem__(self, item: Union[int, slice]):

        # Create the command
        command = ("__getitem__", self.id_at_location, [item], {})

        # Send the command
        obj_id = self.owner.send_command(self.location, command)

        # if item is of int, that means we queried a token rather than a span
        # so we create a TokenPointer
        if isinstance(item, int):

            # create a TokenPointer from obj_id
            token = TokenPointer(location=self.location, id_at_location=obj_id, owner=self.owner)

            return token

        # This means item was of type slice
        # so we create a SpanPointer from the obj_id
        span = SpanPointer(location=self.location, id_at_location=obj_id, owner=self.owner)

        return span

    def as_doc(self):
        """Create a `Doc` object with a copy of the `Span`'s tokens.

        Returns:
            The new `Doc` copy (or id to `Doc` object) of the span.
        """
        # Avoid circular imports
        from .doc_pointer import DocPointer

        # Create the command
        command = ("as_doc", self.id_at_location, [], {})

        # Send the command
        doc_id = self.owner.send_command(self.location, command)

        # Create a DocPointer from doc_id
        doc = DocPointer(location=self.location, id_at_location=doc_id, owner=sy.local_worker)

        return doc
