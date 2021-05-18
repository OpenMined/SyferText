class TextDatasetMeta:
    """Provides meta information about a language modeling dataset
    locally stored as text files.
    Objects of this class are created by data owners and sent to
    the grid of workers to be searchable by data scientists.
    """

    def __init__(self, train_path: str, valid_path: str = None, test_path: str = None):

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
