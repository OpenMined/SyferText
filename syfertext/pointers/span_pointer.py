from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import syft as sy

from typing import List
from typing import Union
from ..typecheck.typecheck import type_hints


class SpanPointer(ObjectPointer):
    """An Object Pointer that points to the Span Object at remote location"""

    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
    ):
        """Create a Span Pointer from `location` where the `Span` object resides and
        `id_at_location`, the id of the `Span` object at that location. 

        Args:
            location (BaseWorker): the worker where the `Span` object resides that this
                SpanPointer will point to.
            
            id_at_location (int or str): the id of the `Span` object at the `location` worker.
            
            owner (BaseWorker): the owner of the this object ie. `SpanPointer`

        Returns:
            A `SpanPointer` object
        """

        super(SpanPointer, self).__init__(
            location=location, id_at_location=id_at_location, owner=owner, id=id,
        )

    @type_hints
    def __len__(self) -> int:

        # Send the command
        length = self.owner.send_command(
            recipient=self.location, cmd_name="__len__", target=self, args_=tuple(), kwargs_={}
        )

        return length

    @type_hints
    def __getitem__(self, item: Union[slice, int]) -> "Span":

        # if item is int, so we are trying to access to token
        assert isinstance(
            item, slice
        ), "You are not authorised to access a `Token` from a `SpanPointer`"

        # Send the command
        obj_id = self.owner.send_command(
            recipient=self.location, cmd_name="__getitem__", target=self, args_=(item,), kwargs_={}
        )

        # we create a SpanPointer from the obj_id
        span = SpanPointer(location=self.location, id_at_location=obj_id, owner=self.owner)

        return span

    @type_hints
    def as_doc(self) -> "Doc":
        """Create a `Doc` object with a copy of the `Span`'s tokens.

        Returns:
            The new `Doc` copy (or id to `Doc` object) of the span.
        """
        # Avoid circular imports
        from .doc_pointer import DocPointer

        # Send the command
        doc_id = self.owner.send_command(
            recipient=self.location, cmd_name="as_doc", target=self, args_=tuple(), kwargs_={}
        )

        # Create a DocPointer from doc_id
        doc = DocPointer(location=self.location, id_at_location=doc_id, owner=sy.local_worker)

        return doc
