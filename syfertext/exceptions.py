from syft.generic.pointers.string_pointer import StringPointer

import syfertext
from .doc import Doc
from syfertext.pointers.doc_pointer import DocPointer

from typing import Union


class ObjectNotCallableError(Exception):
    """Exception raised when a wrong object is added to pipeline
    This exception is raised when an object which doesn't
    has a __call__ attribute is added to the pipeline
    by passing as a parameter in add_pipe method
    """

    def __init__(self, param_name):
        message = (
            "Passed in object for argument "
            + param_name
            + " is not a callable."
            + "Please provide a callable object."
        )
        super().__init__(message)


class DuplicateNameError(Exception):
    """Exception raised when pipeline component with same name is already present in pipeline
    This exception is raised when a component with name similar
    to one already available in pipeline is added to the pipeline
    """

    def __init__(self, name):
        message = (
            "Pipeline component with name:"
            + name
            + " already exists in pipeline."
            + "Please use a different name."
        )
        super().__init__(message)


class InvalidPositionError(Exception):
    """Exception raised when position of component in pipeline is ambiguous
    """

    def __init__(self, **kwargs):
        message = (
            "Component can not be added to pipeline. "
            + "You have provided arguments values "
            + str(kwargs)
            + " which makes the position of new component ambiguous."
            + "Please provide a valid set of arguments."
        )
        super().__init__(message)


class PipelineComponentNotFoundError(Exception):
    """Exception raised when the specified component is not found in pipeline
    """

    def __init__(self, name):
        message = (
            "No component with specified name: "
            + name
            + " was found in Pipeline."
            + "Please provide name of a component"
            + "available in pipeline"
        )
        super().__init__(message)


class ObjectNotCollocatedError(Exception):
    """Raised when a Pipeline component is called to process a
     DocPointer object pointing to data on a remote machine."""

    def __init__(self, object_name):
        message = (
            "A local `"
            + object_name
            + "` can not operate on a remote Document."
            + " Please set `remote` = True while adding"
            + " it to the pipeline. Since "
            + object_name
            + " object is not a private object, sending"
            + " it to a remote worker is safe."
        )

        super().__init__(message)


class SubPipelineNotCollocatedError(Exception):
    """Raised when a String to be tokenized or Doc object to be modified
    by a SubPipeline object is not on the same machine as SubPipeline object.
    The goal is to provide as useful input as possible to help the user
    identify which objects are where so that they can debug
    which one needs to be moved."""

    def __init__(self, object_a, object_b, attr="a method"):

        # Might need to consider this later
        # if hasattr(object_a, "child") and object_b.is_wrapper:
        #     object_a = object_.child
        #
        # if hasattr(object_b, "child") and object_b.is_wrapper:
        #     object_b = object_b.child

        # Object_a type is used in constructing relevant messages
        if isinstance(object_a, str) or isinstance(object_a, StringPointer):
            obj_a_type = "String"
            action = "tokenize"
        elif isinstance(object_a, Doc) or isinstance(object_a, DocPointer):
            obj_a_type = "Doc"
            action = "modify"
        else:
            obj_a_type = "String/Doc"
            action = "process"

        if (isinstance(object_a, StringPointer) or isinstance(object_a, DocPointer)) and isinstance(
            object_b, syfertext.pipeline.pointers.SubPipelinePointer
        ):
            message = (
                "You tried to "
                + action
                + obj_a_type
                + " object using pipeline by calling"
                + attr
                + " on the SubPipeline object."
                + " But they are not on the same machine!"
                + obj_a_type
                + " object is on"
                + str(object_a.location)
                + " while the SubPipeline component is on "
                + str(object_b.location)
                + ". Use a combination of .move(), .get(), and/or .send() to co-locate them to the same machine."
            )
        elif isinstance(object_a, StringPointer) or isinstance(object_a, DocPointer):
            message = (
                "You tried to "
                + action
                + obj_a_type
                + " object using pipeline by calling"
                + attr
                + " on the SubPipeline object. But "
                + obj_a_type
                + " object is located on another machine (is a"
                + obj_a_type
                + "Pointer"
                + "). Call .get() on the "
                + obj_a_type
                + "Pointer"
                + " or .send("
                + str(object_a.location.id)
                + ") on the SubPipeline component.\n"
                + obj_a_type
                + " object: "
                + str(object_a)
                + "\nSubPipeline component : "
                + str(object_b)
            )
        elif isinstance(object_b, syfertext.pipeline.pointers.SubPipelinePointer):
            message = (
                "You tried to "
                + action
                + obj_a_type
                + " object using pipeline by calling"
                + attr
                + " on the SubPipeline object. But "
                + " SubPipeline component is located on another machine (is a SubPipelinePointer)."
                + " Call .get() on the SubPipelinePointer or .send("
                + str(object_b.location.id)
                + ") on the "
                + obj_a_type
                + " object.\n"
                + obj_a_type
                + " object: "
                + str(object_a)
                + "\nSubPipeline component : "
                + str(object_b)
            )
        else:
            message = (
                "You tried to "
                + action
                + obj_a_type
                + " object using pipeline by calling"
                + attr
                + " on the SubPipeline object. But "
                + "they are not on the same machine!!"
                + " Try calling .send(), .move(), and/or .get() on these objects to get them to the same"
                + "worker before calling methods that involve them working together."
            )

        super().__init__(message)
