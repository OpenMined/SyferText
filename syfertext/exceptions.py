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
