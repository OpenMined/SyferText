import torch
import torch.nn

from abc import abstractmethod

class Model(nn.Module):
    """Abstract base class for all downstream task models in SyferText, such as SequenceTagger and TextClassifier.
    Every custom model must implement these methods if you want to replace default pipline models.
    """
    def __init__(self,**kwargs):
        super(Model, self).__init__()
        pass


    @abstractmethod
    def loss(self, batch, compute_predictions=False)-> torch.tensor :
        """
        -to compute negative log likelihood.
        -Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training.
        Parameters:
            batch: A minibatch, instance of torchtext.data.Batch
            compute_predictions: If true compute and provide predictions, else None
        Returns:
            Tuple of loss and predictions
        """
        raise NotImplementedError("Must implement loss()")



    def save(self, model_file: Union[str, Path]):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        torch.save(model_state, str(model_file), pickle_protocol=4)


    @abstractmethod
    def _get_state_dict(self):
        """Returns the state dictionary for this model. Implementing this enables the save() and save_checkpoint()
        functionality."""
        # self.state_dict() (the nn.module function) this is must, but other diffrent 
        # model parmametrs which are essential for creating model must be here
        pass

    @staticmethod
    @abstractmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary."""
        pass

    @classmethod
    def load(cls, model_pt: Union[str, Path]):
        """
        Loads the model from the given file.

        Args:
            model_pt: the path to model file

        Return: 
            model :the loaded model
        """

        state = torch.load(model_pt)

        model = cls._init_model_with_state_dict(state)
        model.eval()
        # Todo : load on device 

        return model



