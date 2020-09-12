import syft as sy
import syft.serde.msgpack.serde as serde
from syft.workers.base import BaseWorker
from syft.generic.abstract.tensor import AbstractTensor
from syft.generic.abstract.sendable import AbstractSendable

from .. import LOCAL_WORKER
from ..doc import Doc
from ..utils import msgpack_code_generator
from ..pointers.doc_pointer import DocPointer


from typing import Dict
from typing import Set
from typing import Union


class AverageDocEncoder(AbstractSendable):
    """This class defines a document encoder consisting of
    computing the average of all token embedding contained
    in a Doc object.

    Excluding specific tokens is supported as well as getting
    encrypted versions of the document average vector.
    """

    def __init__(
        self, excluded_tokens: Dict[str, Set[object]] = None, owner: BaseWorker = None
    ) -> None:
        """Initialize the object

        Args:
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on their custom attribute values.
            owner (optional): The worker on which the object is located.
                The end user does not need to specify this argument. However,
                it is always set by the `detail` method. If no owner is
                found, the global LOCAL_WORKER is used instead. Actually,
                we can alway use LOCAL_WORKER and remove the `owner` arugment
                here, but this won't work for virtual workers where
                `LOCAL_WORKER` does not change from worker to another.

        """

        self.excluded_tokens = excluded_tokens

        # Set the owner
        self.owner = owner if owner is not None else LOCAL_WORKER

        # Initialize the `mpc` property that will hold SMPC config
        # This configuration will include two workers to use to
        # send shares and one crypto provider

        # Initialize to None and only populate it if MPC is possible
        self.mpc = None

    def __call__(
        self, doc: Union[Doc, DocPointer], crypto_config: Dict[str, object] = None
    ) -> Union[sy.AdditiveSharingTensor, AbstractTensor]:
        """Perform encoding of the document.

        Encoding here is performed by computing the average embedding of
        the tokens included in the doc. Some tokens specified in
        `self.excluded_tokens` are excluded.

        If an encryption mode is specified using `encryption`, then the
        average vector is encrypted before being returned.

        Args:
            doc: The Doc object to be encoded or a pointer to it.
            crypto_config: A dictionary containing configuration for the
                encryption scheme chosen.
                for example, if 'mpc' is chosen, the `config` argument
                will have the following format:
                - config['mpc']['node_0'] for the first node to hold
                  the shares.
                - config['mpc']['node_1'] for the second node to hold
                  the shares.
                - config['mpc']['crypto_provider'] for the node that
                  provides crypto primitives.
                - other configurations to be supported in the future.

        Returns:
            A Syft tensor in case no encryption is used. If in 'mpc'
                encryption mode, the tensor returned will be a
                AdditiveSharingTensor.
        """

        # Case 1: SMPC encryption required.
        # In this case, `doc` is a DocPointer object although that it
        # could also be a Doc object in theory.
        # The returned tensor is a Syft AdditiveSharingTensor object.
        if "mpc" in crypto_config:

            # Encrypt the vector
            vector_enc = doc.get_encrypted_vector(
                crypto_config["mpc"]["node_0"],
                crypto_config["mpc"]["node_1"],
                crypto_provider=crypto_config["mpc"]["node_2"],
                requires_grad=False,
                excluded_tokens=self.excluded_tokens,
            )

            # Return the vector
            return vector_enc

        # Case 2: No encryption is required.
        # In the current implementation, we consider that when no
        # encryption is required, the Doc object is on the same
        # node as the AverageDocEncoder object.
        # That is, `doc` is a `Doc` object.
        else:

            # Get the doc vector
            vector = doc.get_vector(excluded_tokens=self.excluded_tokens)

            # Return the vector
            return vector

    @staticmethod
    def simplify(worker: BaseWorker, average_doc_encoder: "AverageDocEncoder") -> tuple:
        """Simplifies a AverageDocEncoder object.

        Args:
            worker (BaseWorker): The worker on which the
                simplify operation is carried out.
            average_doc_encoder: the AverageDocEncoder object
                to simplify.

        Returns:
            (tuple): The simplified AverageDocEncoder object.

        """

        # Simplify the pipeline name
        excluded_tokens_simple = serde._simplify(worker, average_doc_encoder.excluded_tokens)

        return (excluded_tokens_simple,)

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Takes a simplified AverageDocEncoder object, details it
        and returns a AverageDocEncoder object.

        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            simple_obj (tuple): the simplified AverageDocEncoder object.

        Returns:
            (AverageDocEncoder): The AverageDocEncoder object.
        """

        # Unpack the simplified objects
        (excluded_tokens_simple,) = simple_obj

        # Detail the pipeline name
        excluded_tokens = serde._detail(worker, excluded_tokens_simple)

        # Instantiate a AverageDocEncoder object
        average_doc_encoder = AverageDocEncoder(excluded_tokens=excluded_tokens, owner=worker)

        return average_doc_encoder

    @staticmethod
    def get_msgpack_code() -> Dict[str, int]:
        """This is the implementation of the `get_msgpack_code()`
        method required by PySyft's SyftSerializable class.
        It provides a code for msgpack if the type is not present in proto.json.

        The returned object should be similar to:
        {
            "code": int value,
            "forced_code": int value
        }

        Both keys are optional, the common and right way would be to add only the "code" key.

        Returns:
            dict: A dict with the "code" and/or "forced_code" keys.
        """

        # If a msgpack code is not already generated, then generate one
        if not hasattr(AverageDocEncoder, "proto_id"):
            AverageDocEncoder.proto_id = msgpack_code_generator()

        code_dict = dict(code=AverageDocEncoder.proto_id)

        return code_dict
