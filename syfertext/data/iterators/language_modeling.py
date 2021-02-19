import torch


class LMBPTTIterator:
    def __init__(
        self,
        batch_size: int,
        bptt_len: int,
        shuffle: bool = False,
        dataset_reader=None,
        mode="train",
    ):

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.bptt_len = bptt_len
        self.dataset_reader = dataset_reader

    def load(self, dataset_meta):

        # Read the dataset
        self.dataset_reader.read(dataset_meta=dataset_meta)

    def __iter__(self):

        # Reset the iterator
        self.index = 0

        return self

    def __next__(self):

        # Stop iterating if all examples have been loaded
        if self.index == self.num_examples:
            raise StopIteration

        # Load training examples
        batch_examples = []

        for i in range(self.batch_size):

            # Load one example
            example = self._load_example()

            # Add it to the batch
            batch_examples.append(example)

        # Create the batch
        batch = self._collate(batch_examples=batch_examples)

        return batch

    @property
    def num_examples(self):
        """Returns that number of non-overlapping  examples
        in the dataset
        """

        num_examples = (len(self.dataset_reader.examples[0]) - 1) // self.bptt_len

        return num_examples

    @property
    def num_batches(self):
        """Returns the total number of batches. The last batch
        is dropped if its size is less than self.batch_size.
        """

        num_batches = self.num_examples // batch_size

        return num_batches

    def _load_example(self):
        """Load one example from the underlying dataset.
        An example in training mode is a tuple of two tensors:
        The first tensor is the input to the network.
        The second tensor is the input shifted by one word.
        """

        # For convenience, get a reference to the tensor representing
        # the dataset
        dataset = self.dataset_reader.examples[0]

        # Get the input tensor
        inpt = dataset.narrow(
            dim=0, start=self.index * self.bptt_len, length=self.bptt_len
        )

        # Get the target tensor
        target = dataset.narrow(
            dim=0, start=self.index * self.bptt_len + 1, length=self.bptt_len
        )

        # Increment the index
        self.index += 1

        return inpt, target

    def _collate(self, batch_examples):
        """Take a list of training examples and
        transform it to a batch
        """

        # Create the input_batch and the target batch.
        # Each will be of size (bptt_len, batch_size)
        input_batch, target_batch = list(
            map(lambda x: torch.stack(x).transpose(0, 1), zip(*batch_examples))
        )

        return input_batch, target_batch
