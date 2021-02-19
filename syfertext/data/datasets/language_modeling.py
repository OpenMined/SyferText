# Syfertext relative
from ..dataset_metas import LanguageModelingDatasetMeta
from ..readers.language_modeling import LanguageModelingDatasetReader

# Third party
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """This class is responsible of reading a language modeling
    dataset and preparing it for data loaders.
    """

    def __init__(self, encoder=None, mode=None):
        pass

    def __call__(self, dataset_meta: LanguageModelingDatasetMeta):

        # Create a dataset reader and read the dataset
        dataset_reader = LanguageModelingDatasetReader(
            dataset_meta=dataset_meta, tokenizer=encoder.tokenizer, mode=mode
        )

        # Read the dataset splits as lists of tokens
        # The returned value is a dict of three values for the train, val and test
        # splits if no mode is specified. If mode is equal to 'train', 'val' or 'test'
        # then only the corresponding split is returned
        self.splits = dataset_reader.read()

        # If encoder does not have a vocabulary, create one
        if not encoder.has_vocab and mode in [None, "train"]:

            # Create the vocabulary
            # If this is part of a federated learning job
            # the encoder will know that creating the vocab involves other workers
            encoder.create_vocab(self.splits["train"])
