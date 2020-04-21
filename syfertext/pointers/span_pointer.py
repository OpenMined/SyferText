from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import syft as sy

from typing import List
from typing import Union


class SpanPointer(ObjectPointer):
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

        super(SpanPointer, self).__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=garbage_collect_data,
            tags=tags,
            description=description,
        )

    def __len__(self):

        # Create the command
        command = ("__len__", self.id_at_location, [], {})

        # Send the command
        length = self.owner.send_command(self.location, command)

        return length

    def __getitem__(self, item):

        assert isinstance(item, slice), (
            "SpanPointer object can't return a Token. Please call"
            "__getitem__ on a slice to get a pointer to a Span residing"
            "on remote machine."
        )

        # Create the command
        command = ("__getitem__", self.id_at_location, [item], {})

        # Send the command
        span_id = self.owner.send_command(self.location, command)

        # Create a SpanPointer from the span_id
        span = SpanPointer(location=self.location, id_at_location=span_id, owner=sy.local_worker)

        return span
