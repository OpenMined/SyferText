# Syfertext relative
from ..meta import LMDatasetMeta
from ..readers.language_modeling import LMDatasetReader

# Third party
from torch.utils.data import Dataset


class LanguageModelingDataset(Dataset):
    """This class is responsible of reading a language modeling
    dataset and preparing it for data loaders.

    IMPORTANT: Building a vocab should be done beforehand in the
    language object before deploying subpipelines. It should be
    a separate operation that sends an object called VocabBuilder
    to each data owner and creates a vocab that is then used for
    the encoder.
    """

    def __init__(self, encoder=None, mode=None):

        self.encoder = encoder
        self.mode = mode

    def __call__(self, dataset_meta: LMDatasetMeta):

        # Create a dataset reader and read the dataset
        dataset_reader = LMDatasetReader(
            dataset_meta=dataset_meta, encoder=self.encode, mode=self.mode
        )

        # Read the dataset for the requested mode
        # 'train', 'valid' or 'test'
        self.examples = dataset_reader.read()
