from ..state import State
from ..doc import Doc
from .average_doc_encoder import AverageDocEncoder
from .. import LOCAL_WORKER
from ..utils import msgpack_code_generator
from ..utils import create_mpc_config
from ..utils import create_state_query
from ..utils import search_resource
from ..pointers.doc_pointer import DocPointer
from ..pointers import StatePointer


import torch as th

import syft as sy
from syft.generic.abstract.sendable import AbstractSendable
from syft.serde.syft_serializable import SyftSerializable
import syft.serde.msgpack.serde as serde
from syft.serde.msgpack.serde import msgpack_global_state
from syft.workers.base import BaseWorker

import collections

from typing import Dict
from typing import Set
from typing import List
from typing import Union


class SingleLabelClassifier(AbstractSendable):
    """This class bundles a PySyft plan that acts like a single label
    classifier. The plan should be able to take a tensor representing one
    sentence (batch size is 1). It then produces a prediction as
    a logit vector.

    This class also takes a document encoder, which is an object that takes a
    Doc object and produces the plan's input vector.
    """

    def __init__(
        self,
        doc_encoder: AverageDocEncoder = None,
        classifier: sy.Plan = None,
        encryption: str = None,
        labels: List[str] = None,
        logits_index: int = 0,
    ):
        """Initializes the SingleLabelClassifier object.

        Args:
            doc_encoder: The object that takes a Doc or a DocPointer and
                produces the model's input.
            classifier: The PySyft Plan that performs the classification job.
                its output should be the logits vector.
            encryption: The encryption scheme to use. Only 'mpc' is currently supported.
            labels: A list of strings to hold the textual name of each class label.
                The index of each label is used by the classifier as the class label.
            logits_index: This it the index of the logits tensor in the output

        """

        super(SingleLabelClassifier, self).__init__()

        # Do some object setup
        self._setup(
            doc_encoder=doc_encoder,
            classifier=classifier,
            encryption=encryption,
            labels=labels,
            logits_index=logits_index,
        )

    def _setup(
        self,
        doc_encoder: AverageDocEncoder,
        classifier: sy.Plan,
        encryption: str,
        labels: List[str],
        logits_index: int,
    ):
        """Performs the real object initialization. Object
        properties are set here and classifier is prepared
        (including decryption  and building the Syft Plan if
        required).

        The reason we separate initialization in this method
        is that initialization might happen in two different
        places:
            1. In the `__init__()` method when the object is
                first created.
            2. In the `load_state()` method when the object
                has been already deployed and requested for
                for inference.

        Args:
            doc_encoder: The object that takes a Doc or a DocPointer and
                produces the model's input.
            classifier: The PySyft Plan that performs the classification job.
                its output should be the logits vector.
            encryption: The encryption scheme to use. Only 'mpc' is currently supported.
            labels: A list of strings to hold the textual name of each class label.
                The index of each label is used by the classifier as the class label.


        Modifies:
            The following properties are created: `self.doc_encoder`,
                `self.classifier`, `self.encryption`, `self.labels`,
                `self.logits_index`
        """

        # Set some object properties
        self.doc_encoder = doc_encoder
        self.classifier = classifier
        self.encryption = encryption
        self.labels = labels
        self.logits_index = logits_index

        # Create the configuration dict.
        # This dict would hold the encryption configuration if one
        # is required, or None otherwise
        self._create_config()

        # Prepare the classifier's Syft Plan. Check the method's
        # docstring to know more.
        self._prepare_classifier()

        # If encryption is required, encrypt the classifier.
        # The reason encryption is performed again here is in case
        # the classifier is going to be used locally before
        # deployement to PyGrid.
        if self.encryption:
            self._encrypt_classifier()

    def _prepare_classifier(self) -> None:
        """Prepares the classifier to being deployed or reencrypted.
        If the classifier's Syft Plan is not built, then build it.
        If the classifier is encrypted, decrypt it.

        Modifies:
            self.classifier: The classifier's Syft plan  is decrypted
                and built.
        """

        # If the classifier is not yet loaded, return.
        # This happens when the classifier's object is created
        # on a worker after deployment and before the state is
        # loaded.
        if not isinstance(self.classifier, sy.Plan):
            return

        # Detect the input vector's number of features
        # This will be used to build the classifier's Syft Plan
        num_features = self.classifier.parameters()[0].data.shape[1]

        # Detect whether the vector is 'mpc' encrypted.
        # If it is the case, then decrypt it
        # There is a better way maybe to detect this?
        if isinstance(self.classifier.parameters()[0].data, sy.FixedPrecisionTensor):
            self.classifier.get().float_prec()

        # Build the plan if it is not yet built
        if not self.classifier.is_built:
            self.classifier.build(th.ones(1, num_features))

    def _create_config(self) -> None:
        """Create configurations for the encryption scheme to be used.

        Modifies:
            `self.config` which is a dict that holds
                configurations for the crypto scheme to be used.
                Only 'mpc' config is currently supported.
        """

        # Initialize a dictionary to hold the crypto configuration
        self.config = {}

        # SMPC case
        if self.encryption == "mpc":

            # Get the SMPC configuration dictionary
            mpc_config = create_mpc_config(worker=self.owner)

            # If the dictionary is `None`, then MPC is impossible.
            # In this case, the config property should be `None`
            if mpc_config is None:
                return

            # Load the mpc configurations
            self.config["mpc"] = mpc_config

    def _encrypt_classifier(self):
        """Encrypt `self.classifier` using the encryption scheme specified
        in `self.encryption` according to configuration in `self.config`.

        Returns:
            The encrypted classifier.
        """

        # Encrypt the classifier
        if self.encryption == "mpc":
            self.classifier.fix_precision().share(
                self.config["mpc"]["node_0"],
                self.config["mpc"]["node_1"],
                crypto_provider=self.config["mpc"]["node_2"],
            )
            print(self.classifier.parameters()[0])

            self.classifier.parameters()[0] = self.classifier.parameters()[0].send(
                self.config["mpc"]["node_0"]
            )
            self.classifier.parameters()[0].get()

            print(self.classifier.parameters()[0])

    @property
    def pipeline_name(self) -> str:
        """A getter for the `_pipeline_name` property.

        Returns:
           The lower cased `_pipeline_name` property.
        """

        return self._pipeline_name.lower()

    @pipeline_name.setter
    def pipeline_name(self, name: str) -> None:
        """Set the pipeline name to which this object belongs.

        Args:
            name: The name of the pipeline.
        """

        # Convert the name of lower case
        if isinstance(name, str):
            name = name.lower()

        self._pipeline_name = name

    @property
    def name(self) -> str:
        """A getter for the `_name` property.

        Returns:
           The lower cased `_name` property.
        """

        return self._name.lower()

    @name.setter
    def name(self, name: str) -> None:
        """Set the component name.

        Args:
            name: The name of the component
        """

        # Convert the name of lower case
        if isinstance(name, str):
            name = name.lower()

        self._name = name

    @property
    def access(self) -> Set[str]:
        """Get the access rules for this component.

        Returns:
            The set of worker ids where this component's state
            could be sent.
            If the string '*' is included in the set,  then all workers are
            allowed to receive a copy of the state. If set to None, then
            only the worker where this component is saved will be allowed
            to get a copy of the state.
        """

        return self._access_rules

    @access.setter
    def access(self, rules: Set[str]) -> None:
        """Set the access rules of this object.

        Args:
            rules: The set of worker ids where this component's state
                could be sent.
                If the string '*' is included in the set,  then all workers are
                allowed to receive a copy of the state. If set to None, then
                only the worker where this component is saved will be allowed
                to get a copy of the state.
        """

        self._access_rules = rules

    def __call__(self, doc: Union[Doc, DocPointer]) -> Union[Doc, DocPointer]:
        """Perform inference using `self.classifier`. The document is
        first encoded using `self.doc_encoder` according to the selected
        encryption scheme in `self.encryption`, if one is specified.
        Then the encrypted (or not) embedding vector representing the
        document is passed to the classifier.

        In case of encrypted inference, the encrypted logits are securely
        sent to the worker belonging to the data owner, which would have
        necessary authorizations to decrypt the logits and obtain the
        predicted class label.

        Args:
            doc: the document to be used as the input for inference
                or a pointer to it.

        Returns
            The modified document or its pointer. A modified document
                is one that contains the predicted class label resulting
                from the classifier.
        """

        # Encode the document and obain the embedding vector
        doc_vector = self.doc_encoder(doc=doc, crypto_config=self.config)
        print(self.config)
        print(doc_vector)

        # Perform classification and get the logits
        output = self.classifier(doc_vector)

        if isinstance(output, collections.abc.Sequence):
            logits = output[self.logits_index]
        else:
            logits = logits

        # Secure delivery of the predicted label.
        # In case encryption is used, decryption takes place
        # on the worker holding the private data.
        doc.decode_logits(
            task_name=f"{self.pipeline_name}__{self.name}",
            logits=logits,
            labels=self.labels,
            single_label=True,
            encryption=self.encryption,
        )

        return doc

    def dump_state(self) -> State:
        """Returns a State object that holds the current state of this object.

        Returns:
            A State object that holds a simplified version of this object's state.
        """

        # Prepare the classifier.
        # This is necessary in order to decrypt the classifier in case
        # it has been used in encrypted mode before deployment.
        self._prepare_classifier()

        # Simplify the document encoder. First, the proto ID of the object
        # should be obtained. This is important for the detailer to know
        # which detailer method to usefrom PySyft at the remote machine.
        proto_id = self.doc_encoder.get_msgpack_code()["code"]

        # Get a reference to the document encoder object
        encoder = self.doc_encoder

        # Simplify the document encoder
        doc_encoder_simple = (proto_id, encoder.simplify(LOCAL_WORKER, encoder))

        # Simplify the remaining the properties
        classifier_simple = serde._simplify(self.owner, self.classifier)
        encryption_simple = serde._simplify(self.owner, self.encryption)
        labels_simple = serde._simplify(self.owner, self.labels)
        logits_index_simple = serde._simplify(self.owner, self.logits_index)

        # Create the query. This is the ID according to which the
        # State object is searched for on across workers
        state_id = f"{self.pipeline_name}:{self.name}"

        # Create the State object
        state = State(
            simple_obj=(
                doc_encoder_simple,
                classifier_simple,
                encryption_simple,
                labels_simple,
                logits_index_simple,
            ),
            id=state_id,
            access=self.access,
        )

        return state

    def load_state(self) -> None:
        """Search for the state of this object on PyGrid.

        Modifies:
            self.doc_encoder: The `doc_encoder` property is assigned the required
                object.
            self.classifier: The `classifier` object is loaded by assigning a model.

        """

        # Create the query. This is the ID according to which the
        # State object is searched on PyGrid
        state_id = create_state_query(pipeline_name=self.pipeline_name, state_name=self.name)

        # Search for the state
        result = search_resource(query=state_id, local_worker=self.owner)

        # If no state is found, return
        if not result:
            return

        # If a state is found get either its pointer if it is remote
        # or the state itself if it is local
        elif isinstance(result, StatePointer):

            # Get a copy of the state using its pointer
            state = result.get_copy()

        elif isinstance(result, State):
            state = result

        # Get the simplified objects contained in the state
        (
            doc_encoder_simple,
            classifier_simple,
            encryption_simple,
            labels_simple,
            logits_index_simple,
        ) = state.simple_obj

        # Detail the document encoder
        proto_id, doc_encoder_simple = doc_encoder_simple
        doc_encoder = msgpack_global_state.detailers[proto_id](LOCAL_WORKER, doc_encoder_simple)

        # Detail the simple classifier contained in the state
        classifier = serde._detail(self.owner, classifier_simple)
        encryption = serde._detail(self.owner, encryption_simple)
        labels = serde._detail(self.owner, labels_simple)
        logits_index = serde._detail(self.owner, logits_index_simple)

        # Setup the object
        self._setup(
            doc_encoder=doc_encoder,
            classifier=classifier,
            encryption=encryption,
            labels=labels,
            logits_index=logits_index,
        )

    @staticmethod
    def simplify(worker: BaseWorker, single_label_classifier: "SingleLabelClassifier") -> tuple:
        """Simplifies a SingleLabelClassifier object.

        Args:
            worker (BaseWorker): The worker on which the
                simplify operation is carried out.
            single_label_classifier: the SingleLabelClassifier object
                to simplify.

        Returns:
            (tuple): The simplified SingleLabelClassifier object.

        """

        # Simplify the component and the pipeline names
        name_simple = serde._simplify(worker, single_label_classifier.name)
        pipeline_name_simple = serde._simplify(worker, single_label_classifier.pipeline_name)

        return (name_simple, pipeline_name_simple)

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Takes a simplified SingleLabelClassifier object, details it
        and returns a SingleLabelClassifier object.

        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            simple_obj (tuple): the simplified SingleLabelClassifier object.

        Returns:
            (SingleLabelClassifier): The SingleLabelClassifier object.
        """

        # Unpack the simplified objects
        name_simple, pipeline_name_simple = simple_obj

        # Detail the the component and the pipeline names
        name = serde._detail(worker, name_simple)
        pipeline_name = serde._detail(worker, pipeline_name_simple)

        # Instantiate a SingleLabelClassifier object
        single_label_classifier = SingleLabelClassifier()
        single_label_classifier.name = name
        single_label_classifier.pipeline_name = pipeline_name
        single_label_classifier.owner = worker

        return single_label_classifier

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
        if not hasattr(SingleLabelClassifier, "proto_id"):
            SingleLabelClassifier.proto_id = msgpack_code_generator(
                SingleLabelClassifier.__qualname__
            )

        code_dict = dict(code=SingleLabelClassifier.proto_id)

        return code_dict
