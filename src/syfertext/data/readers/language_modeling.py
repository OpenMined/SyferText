import pathlib
import torch


class TextReader:
    def __init__(self, encoder, mode):

        self.encoder = encoder
        self.mode = mode

    def read(self, dataset_meta):
        """Read the dataset of the specified mode, and return
        all of its text as a list of integer indexes.
        """

        # Initialize the list that will hold the encoded text
        encoded_text = []

        # Get the path of the file containing text data
        data_path = getattr(dataset_meta, f"{self.mode}_path")
        data_path = pathlib.Path(data_path)

        # Open the text file to read and encode its text
        with data_path.open() as f:

            # Read all lines
            for line in f.readlines():

                # Encode the line into integer indexes
                enc_output = self.encoder(line)

                # Get the list of token IDs
                line_encoded = enc_output["token_ids"]

                encoded_text.extend(line_encoded)

        # Add the whole dataset as one example
        self.encoded_text = torch.LongTensor(encoded_text)
