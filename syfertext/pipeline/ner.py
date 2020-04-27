from syfertext.doc import Doc
from syfertext.token import Token
from typing import Union

from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde

# Tags
begin_token_tag = "B"
inner_token_tag = "I"
last_token_tag = "L"
unit_entity_token_tag = "U"
non_entity_token_tag = "O"


class EntityRecognizer:
    """This is a very simple model-free EntityRecognizer. It enables to tag specified
       entities in a `Doc` object. The attribute becomes accessible then through the
       Underscore attribute of the`Token` objects in the `Doc`.
    """

    def __init__(self, lookups: dict, default_tag: object = None, case_sensitive: bool = True):
        """Initialize the EntityRecognizer object.

        Args:
            lookups (dict): The keys should be the entity texts to be
                tagged, and values should hold a single tag for each such entity.
                Example: {'Bob': PERSON, 'William Shakespeare' : PERSON, 'New York' : GPE}.
            case_sensitive: (bool, optional): If set to True, then matching
                token texts to `lookups` will become case sensitive.
                Defaults to True.
            default_tag: (object, optional): The default entity tag to be assigned
                in case the token text matches no entry in `lookups`.
        """

        self.default_tag = default_tag

        # Desensitize tokens in lookups if `case_sensitive` is False
        if case_sensitive:
            self.lookups = lookups
        else:
            self.lookups = self._desensitize_lookups(lookups)

        self.case_sensitive = case_sensitive

    def factory(self):
        """Creates a clone of this object.
        This method is used by the SupPipeline class to create
        objects using subpipeline templates.
        """

        return EntityRecognizer(
            lookups=self.lookups, default_tag=self.default_tag, case_sensitive=self.case_sensitive
        )

    def __call__(self, doc: Doc):

        # Start tagging entities
        begin_token = False
        prev_tag = self.default_tag

        for token in doc:
            # Get the tag
            tag = self._get_tag(token)

            # Set tokens `ent_type_` attribute to tag
            token.set_attribute(name="ent_type_", value=tag)

            # Token is a part of an entity
            if tag != self.default_tag:

                # This is the first token of the current entity
                if not begin_token or tag != prev_tag:

                    # Set tokens `ent_iob_` attribute to 'B'
                    token.set_attribute(name="ent_iob_", value=begin_token_tag)

                    # Indicate token part of an entity has been encountered
                    begin_token = True

                # Token is not the first token of the current entity
                else:

                    # Set tokens `ent_iob_` attribute to 'I'
                    token.set_attribute(name="ent_iob_", value=inner_token_tag)

            # Token is not a part of an entity
            else:

                # Set tokens `ent_iob_` attribute to 'O'
                token.set_attribute(name="ent_iob_", value=non_entity_token_tag)

            # update prev_tag
            prev_tag = tag

        return doc

    def _get_tag(self, token: Token):
        """Gets the tag that should be assigned to the Token object `token`.
           If no value is found, self.default is returned instead

        Args:
            token (Token): The Token object to which the tag is to be assigned.

        Returns:
            tag (object): Check out the docstring of `__init__()`.
        """

        # Get the token text
        token_text = token.text if self.case_sensitive else token.text.lower()

        tag = self.lookups.get(token_text, self.default_tag)

        return tag

    @staticmethod
    def _desensitize_lookups(lookups: Union[dict, list, set]):
        """Converts every token in `self.lookups` to lower case to enable
           case in-sensitive matching

        Args:
            lookups (set, list or dict): Check out the docstring of `__init__()`.


        Returns:
            A transformed version  of `lookup` where all token texts are in
            lower case.

        """

        return {entity.lower(): lookups[entity] for entity in lookups}

    @staticmethod
    def simplify(worker: BaseWorker, entity_recognizer: "EntityRecognizer"):
        """Simplifies a EntityRecognizer object.

        Args:
            worker (BaseWorker): The worker on which the
                simplify operation is carried out.
            entity_recognizer (EntityRecognizer): The EntityRecognizer object
                to simplify.

        Returns:
            (tuple): The simplified EntityRecognizer object.

        """

        # Simplify the object properties
        lookups = serde._simplify(worker, entity_recognizer.lookups)
        default_tag = serde._simplify(worker, entity_recognizer.default_tag)
        case_sensitive = serde._simplify(worker, entity_recognizer.case_sensitive)

        return (lookups, default_tag, case_sensitive)

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Takes a simplified EntityRecognizer object, details it
           and returns a EntityRecognizer object.

        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            simple_obj (tuple): the simplified SubPipeline object.
        Returns:
            (EntityRecognizer): The EntityRecognizer object.
        """

        # Unpack the simplified object
        lookups, default_tag, case_sensitive = simple_obj

        # Detail each property
        lookups = serde._detail(worker, lookups)
        default_tag = serde._detail(worker, default_tag)
        case_sensitive = serde._detail(worker, case_sensitive)

        # Instantiate a EntityRecognizer object
        entity_recognizer = EntityRecognizer(
            lookups=lookups, default_tag=default_tag, case_sensitive=case_sensitive
        )

        return entity_recognizer
