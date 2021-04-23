from typing import Dict, List
from torch import LongTensor
from transformers import DataCollatorForLanguageModeling


class BERTIterator:
    
    def __init__(self, dataset_reader, batch_size: int, sentence_len: int):
        self.dataset_reader = dataset_reader
        self.batch_size = batch_size
        self.sentence_len = sentence_len

    def load(self, dataset_meta):
        self.dataset_reader.read(dataset_meta)

    def __iter__(self):

        self.index = 0

        return self

    def __next__(self):

        batch_examples = []

        for i in range(self.batch_size):
            example = self._load_example()
            batch_examples.append(example)

        batch = self._collate(batch_examples=batch_examples)

        return batch

    @property
    def num_examples(self):
        """Returns that number of non-overlapping  examples
        in the dataset
        """

        num_examples = len(self.dataset_reader.encoded_text) // self.sentence_len

        return num_examples

    @property
    def num_batches(self):
        """Returns the total number of batches. The last batch
        is dropped if its size is less than self.batch_size.
        """

        num_batches = self.num_examples // self.batch_size

        return num_batches

    def __len__(self):
        return self.num_batches

    def _load_example(self) -> LongTensor:

        # LongTensor containing the dataset
        dataset = self.dataset_reader.encoded_text

        #Getting an example - sequence of length 'sentence_len'
        example = dataset.narrow(
            dim=0, start=self.index * self.sentence_len, length=self.sentence_len
        )

        self.index += 1

        return example

    def _collate(self, batch_examples: List) -> Dict:

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.dataset_reader.encoder.tokenizer_ref,
            mlm = True,
            mlm_probability = 0.15)
        
        return data_collator(batch_examples)


