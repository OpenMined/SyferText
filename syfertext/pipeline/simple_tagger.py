from syfertext.doc import Doc
from syfertext.token import Token
from ..state import State
from ..pointers import StatePointer
from ..utils import msgpack_code_generator
from ..utils import create_state_query
from ..utils import search_resource
from .. import LOCAL_WORKER

from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde
from syft.generic.abstract.sendable import AbstractSendable

from typing import Union
from typing import Dict


class SimpleTagger(AbstractSendable):
    """This is a very simple token-level tagger. It enables to tag specified
    tokens in a `Doc` object. By tagging a token, we mean setting a new
    attribute to that token which holds the desired tag as its value.
    The attribute becomes accessible then through the Underscore attribute
    of the `Token` object.
    """

    def __init__(
        self,
        attribute: str = None,
        lookups: Union[set, list, dict] = None,
        tag: object = None,
        default_tag: object = None,
        case_sensitive: bool = True,
        owner=None,
        model_name=None,
    ):
        """Initialize the SimpleTagger object.


        Args:
            attribute (str): The name of the attribute that will hold the tag.
                this attribute will be accessible through the attribute
                `._` of Token objects. Example `token_object._.<attribute>
            lookups (set, list or dict): If of type `list` of `set`, it should contain
                the tokens that are to be searched for and tagged in the Doc
                object's text. Example: ['the', 'myself', ...]
                If of type `dict`, the keys should be the tokens texts to be
                tagged, and values should hold a single tag for each such token.
                Example: tagging stop words {'the': True, 'myself' : True}.
            tag (object, optional): If `lookups` is of type `list`, then this
                will be the tag assigned to all matched tokens. It will be
                ignored if `lookups` if of type `dict`.
            default_tag: (object, optional): The default tag to be assigned
                in case the token text maches no entry in `lookups`.
            case_sensitive: (bool, optional): If set to True, then matching
                token texts to `lookups` will become case sensitive.
                Defaults to True.
        """
        self.model_name = model_name
        self.owner = owner
        self.attribute = attribute

        # Desensitize tokens in lookups if `case_sensitive` is False
        if case_sensitive:
            self.lookups = lookups
        else:
            self.lookups = self._desensitize_lookups(lookups)

        # If `lookups` is a `list`, convert it to a `set`
        self.lookups = set(self.lookups) if isinstance(self.lookups, list) else self.lookups

        self.case_sensitive = case_sensitive
        self.tag = tag
        self.default_tag = default_tag

    def factory(self):
        """Creates a clone of this object.
        This method is used by the SupPipeline class to create
        objects using subpipeline templates.
        """

        return SimpleTagger(
            attribute=self.attribute,
            lookups=self.lookups,
            tag=self.tag,
            default_tag=self.default_tag,
            case_sensitive=self.case_sensitive,
        )

    def __call__(self, doc: Doc):

        # Start tagging
        for token in doc:

            # Get the tag
            tag = self._get_tag(token)

            # Set the attribute of each matched token to the tag
            token.set_attribute(name=self.attribute, value=tag)

        return doc

    def _desensitize_lookups(self, lookups: Union[dict, list, set]):
        """Converts every token in `self.lookups` to lower case to enable
           case in-sensitive matching

        Args:
            lookups (set, list or dict): Check out the docstring of `__init__()`.


        Returns:
            A transformed version  of `lookup` where all token texts are in
            lower case.

        """

        # Replace dict keys with lower-case versions
        if isinstance(lookups, dict):
            return {token.lower(): lookups[token] for token in lookups}

        # Convert all list or set elements to lower case
        if isinstance(lookups, list) or isinstance(lookups, set):
            return {token.lower() for token in lookups}

    def _get_tag(self, token: Token):
        """Gets the tag that should be assigned to the Token object `token`.
           If now value is found, self.default is used instead


        Args:
            token (Token): The Token object to which the tag is to be assigned.

        Returns:
            tag (object): Check out the docstring of `__init__()`.
        """

        # Get the token text
        token_text = token.text if self.case_sensitive else token.text.lower()

        # If `self.lookups` is a dict, get the corresponding tag
        if isinstance(self.lookups, dict):

            # Get the associated tag
            tag = self.lookups.get(token_text, self.default_tag)

        # If `self.lookups` is a set, use the `self.tag` attribute.
        elif isinstance(self.lookups, set):

            tag = self.tag if token_text in self.lookups else self.default_tag
        return tag

    @staticmethod
    def simplify(worker: BaseWorker, simple_tagger: "SimpleTagger"):
        """Simplifies a SimpleTagger object.

        Args:
            worker (BaseWorker): The worker on which the
                simplify operation is carried out.
            simple_tagger (SimpleTagger): the SimpleTagger object
                to simplify.

        Returns:
            (tuple): The simplified SimpleTagger object.

        """

        # Simplify the object properties
        attribute = serde._simplify(worker, simple_tagger.attribute)
        lookups = serde._simplify(worker, simple_tagger.lookups)
        tag = serde._simplify(worker, simple_tagger.tag)
        default_tag = serde._simplify(worker, simple_tagger.default_tag)
        case_sensitive = serde._simplify(worker, simple_tagger.case_sensitive)
        model_name = serde._simplify(worker, simple_tagger.model_name)

        return (attribute, lookups, tag, default_tag, case_sensitive, model_name)

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Takes a simplified SimpleTagger object, details it
           and returns a SimpleTagger object.

        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            simple_obj (tuple): the simplified SubPipeline object.
        Returns:
            (SimpleTagger): The SimpleTagger object.
        """

        # Unpack the simplified object
        attribute, lookups, tag, default_tag, case_sensitive, model_name = simple_obj

        # Detail each property
        attribute = serde._detail(worker, attribute)
        lookups = serde._detail(worker, lookups)
        tag = serde._detail(worker, tag)
        default_tag = serde._detail(worker, default_tag)
        case_sensitive = serde._detail(worker, case_sensitive)
        model_name = serde._detail(worker, model_name)

        # Instantiate a SimpleTagger object
        simple_tagger = SimpleTagger(
            attribute=attribute,
            lookups=lookups,
            tag=tag,
            default_tag=default_tag,
            case_sensitive=case_sensitive,
            model_name=model_name,
            owner=worker,
        )

        return simple_tagger

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
        # the code is hash of class name
        if not hasattr(SimpleTagger, "proto_id"):
            SimpleTagger.proto_id = msgpack_code_generator(SimpleTagger.__qualname__)

        code_dict = dict(code=SimpleTagger.proto_id)

        return code_dict

    def set_model_name(self, model_name: str) -> None:
        """Set the language model name to which this object belongs.

        Args:
            name: The name of the language model.
        """

        self.model_name = model_name

    def load_state(self, name=None) -> None:
        """Search for the state of this object on PyGrid.

        Modifies:

        """
        if name:
            self.state_name = name
        else:
            self, state_name = self.__class__.__name__.lower()

        # Create the query. This is the ID according to which the
        # State object is searched on PyGrid
        state_id = create_state_query(model_name=self.model_name, state_name=self.state_name)

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

        # Detail the simple object contained in the state
        (
            attribute_simple,
            lookups_simple,
            tag_simple,
            default_tag_simple,
            case_sensitive_simple,
        ) = state.simple_obj

        self.attribute = serde._detail(LOCAL_WORKER, attribute_simple)
        self.lookups = serde._detail(LOCAL_WORKER, lookups_simple)
        self.tag = serde._detail(LOCAL_WORKER, tag_simple)
        self.default_tag = serde._detail(LOCAL_WORKER, default_tag_simple)
        self.case_sensitive = serde._detail(LOCAL_WORKER, case_sensitive_simple)

        """# Load the state
        self.attribute = attribute, self.lookups = lookups,
        self.tag = self.tag, self.default_tag = self.default_tag, self.case_sensitive = case_sensitive
        """

    def dump_state(self, name=None) -> State:
        """Returns a State object that holds the current state of this object.

        Returns:
            A State object that holds a simplified version of this object's state.
        """
        if name:
            self.state_name = name
        else:
            self, state_name = self.__class__.__name__.lower()

        # Simply the state variables
        attribute_simple = serde._simplify(LOCAL_WORKER, self.attribute)
        lookups_simple = serde._simplify(LOCAL_WORKER, self.lookups)
        tag_simple = serde._simplify(LOCAL_WORKER, self.tag)
        default_tag_simple = serde._simplify(LOCAL_WORKER, self.default_tag)
        case_sensitive_simple = serde._simplify(LOCAL_WORKER, self.case_sensitive)

        # Create the query. This is the ID according to which the
        # State object is searched for on across workers
        state_id = f"{self.model_name}:{self.state_name}"

        # Create the State object
        state = State(
            simple_obj=(
                attribute_simple,
                lookups_simple,
                tag_simple,
                default_tag_simple,
                case_sensitive_simple,
            ),
            id=state_id,
            access={"*"},
        )

        return state
