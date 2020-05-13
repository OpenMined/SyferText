# This functionality can be added to can be added to PySyft BaseWorker.

from abc import abstractmethod
from syft.workers.base import BaseWorker
from syfertext.pointers import DocPointer
from syfertext.encdec import encrypt, decrypt

import binascii
import hashlib
import os
import random
from typing import Union
from typing import List
from typing import Dict
from typing import Set


class ExtendedBaseWorker(BaseWorker):
    """Extends the PySyft BaseWorker. Adds the functionality to share public key
    and generate private and secret key. Thus enabling it to participate in
    Diffie-Hellman key exchange protocol.
    It can also act as a SecureWorker and perform the DH key exchange protocol.
    """

    def __init__(
        self,
        hook: "FrameworkHook",
        id: Union[int, str] = 0,
        data: Union[List, tuple] = None,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        auto_add: bool = True,
        message_pending_time: Union[int, float] = 0,
    ):
        super().__init__(
            hook=hook,
            id=id,
            data=data,
            is_client_worker=is_client_worker,
            log_msgs=log_msgs,
            verbose=verbose,
            auto_add=auto_add,
            message_pending_time=message_pending_time,
        )

        # TODO: Make secret and private key's private variables
        self.private_key = None
        self.secret = None
        self.secret_key = None

        # Should I create a class, inheriting set and AbstractObject?
        self.vocab = set()

        # Register it on the object store ?
        # self.register_obj(obj=self.vocab_set)

    def get_enc_vocab(self):
        """Get Encrypted Vocabulary of worker.

        Returns:
            enc_vocab (set): Vocabulary of the worker encrypted using it' secret key.
        """
        # TODO: I guess it will not work out of the box with Grid workers.
        # TODO: Shuffle the vocabulary, before sending.
        enc_vocab = set()

        for token in self.vocab:
            # convert hash value to string
            token = str(token)
            enc_token = encrypt(token, self.secret_key)
            enc_vocab.add(enc_token)
        return enc_vocab

    def receive_indexed_vocab(self, indexed_vocab):
        """Map vocabulary token to indices.

        Args:
            indexed_vocab (dict): Dictionary containing keys as encrypted tokens
                and value as it's index.
        """

        key = self.secret_key

        assert key is not None, ()

        self.indexed_vocab = dict()

        for enc_token, index in indexed_vocab.items():

            # Decrypt token and convert it from bytes to utf-8 encoding
            dec_token_hash = int(decrypt(enc_token, key).decode("utf-8"))

            # map hash to index
            self.indexed_vocab[dec_token_hash] = index

    def generate_private_key(self, bytes_len):
        """Generates a private key.

        Args:
            bytes_len (int): Bytes length of the private key
        """

        # TODO: Check private key is smaller than shared_prime
        _rand = 0
        while _rand.bit_length() < bytes_len:
            # Generate string of random bytes of size bytes_len
            hex_key = binascii.hexlify(os.urandom(bytes_len))
            _rand = int(hex_key, 16)

        self.private_key = _rand

    def generate_public_key(self, shared_prime, shared_base):
        """ Generates and returns a public key.

        Args:
            shared_prime (int): The prime number in DHKE
            shared_base (int): The generator in DHKE
        """
        assert self.private_key is not None, (
            f"{self.id} has no private key." "Please generate private key first."
        )

        # public_key = ( shared_base ^ private_key ) % shared_prime
        public_key = pow(shared_base, self.private_key, shared_prime)
        return public_key

    def generate_secret_key(self, shared_prime, received_public_key):
        """ Generates a secret key for the worker using the received_public_key.
        """

        assert self.private_key is not None, (
            f"{self.id} has no private key." "Please generate private key first."
        )

        # secret = ( public_key ^ private_key ) % shared_prime
        self.secret = pow(received_public_key, self.private_key, shared_prime)

        # Convert the secret to secret key of type bytes and lengths 32
        secret_bytes = bytes(str(self.secret), "utf-8")
        key = hashlib.sha256()
        key.update(secret_bytes)
        self.secret_key = key.digest()

    @staticmethod
    def execute_dh_key_exchange(shared_prime, shared_base, workers: List["ExtendedBaseWorker"]):
        """Executes the Multi-party Diffie-Hellman key exchange protocol.
        At the end of the protocol each worker has a secret key.

        Please refer to the "Operation with more than two parties" section on
        https://en.wikipedia.org/wiki/Diffie–Hellman_key_exchange to understand
        the process in detail.

        Args:
            shared_prime: The public prime number. Should be very large to ensure the strength
                            of encryption.
            shared_base: The public base integer, typically values 2 and 3 can be used.
            workers (List): List of BaseWorkers involved in the key exchange process
        """

        num_workers = len(workers)

        # Generate private key for each worker
        for worker in workers:
            # Generate length of private key in bytes
            key_len = random.randint(1, 128) % 64
            worker.generate_private_key(key_len)

        # If there are n workers, at max n-1 exchanges are needed before
        # generating the secret key for a worker.
        max_exchanges = num_workers - 1

        for i in range(num_workers):

            pointer = i

            # Generate the first public key
            public_key = workers[pointer].generate_public_key(shared_prime, shared_base)

            # Keep track of exchanges performed
            exchanges = 1

            while exchanges < max_exchanges:
                # Update pointer in circular manner
                pointer = (pointer + 1) % num_workers

                # Keep passing the public key from one worker to another as the shared_base
                public_key = workers[pointer].generate_public_key(
                    shared_prime, shared_base=public_key
                )

                # Update number of exchanges
                exchanges += 1

            # Update the pointer in circular manner
            pointer = (pointer + 1) % num_workers

            # Generate the secret key for the worker at the end of the cycle
            workers[pointer].generate_secret_key(shared_prime, received_public_key=public_key)

    def create_vocabulary(
        self, dataset: List[Dict], key: str, excluded_tokens: Dict[str, Set[object]] = None
    ):
        """Combines the vocabulary of each worker to create a new vocabulary `combined_vocab`.
        Please, try to execute this function only once, as it is has a high communication
        overhead.

        TODO: Measure the communication overhead due to this function.

        Args:
            dataset (List): List of dictionaires, where each dictionary is a single example
            key (str): The key in dictionary, which maps to DocPointer
            excluded_tokens (dict): Tokens which need not be assigned indices.

                # TODO: Remove this constraint ?
                Examples:
                    If you are sure, you are not going to use stop words anywhere in your application.
                    Then you can exclude those tokens from being assigned an index. This helps to avoid unnecessary
                    communication overhead.

                    NOTE: If you aren't sure about excluding certain tokens, then better assign them indices
                    so that later if required you don't have to execute the function again.

        Returns:
            vocab_size (int): The size of vocabulary across all workers, which is size of `combined_vocab`.
        """

        combined_vocab = set()

        # Dict mapping worker to it's vocab
        # Helps avoiding calling worker.get_vocab() again
        # while assigning indices
        worker_to_vocab = dict()

        # A list of all workers encountered in the dataset
        workers = set()

        for example in dataset:

            doc = example[key]

            # Add doc's tokens to worker's vocabulary
            doc.add_tokens_to_vocab(excluded_tokens)

            workers.add(doc.location)

        for worker in workers:

            # Get workers vocabulary
            vocab = worker.get_enc_vocab()
            worker_to_vocab[worker] = vocab

            # Take union with `combined_vocab` set
            combined_vocab = combined_vocab.union(vocab)

        vocab_size = len(combined_vocab)

        self._return_indexed_vocab(combined_vocab, worker_to_vocab)

        return vocab_size

    def _return_indexed_vocab(self, combined_vocab, worker_to_vocab):
        """Maps each token in `combined_vocab` to a index. Returns a dictionary to each worker
        mapping its vocab to indices.

        Args:
            combined_vocab (set): The combined vocabulary across all workers
            worker_to_vocab (Dict): Dict containing keys as workers and values their corresponding vocabularies.

        Examples:
            Bob.vocab = ["private", "and", "secure", "nlp"]
            Alice.vocab = ["keeps", "your", "data", "private", "and", "secure"]
            combined_vocab = ["and", "data", "keeps", "nlp", "secure", "private", "your"]

            token_to_index = {"and": 0, "data": 1, "keeps": 2, "nlp": 3, "secure": 4, "private": 5, "your": 6}

            Bob gets back  : {"private": 5, "and": 0, "secure": 4, "nlp": 3}
            Alice gets back: {"keeps": 2, "your": 6, "data": 1, "private": 5, "and": 0, "secure": 4}
        """
        # Assign indices to each token in vocab
        token_to_index = dict()

        for index, token in enumerate(combined_vocab):
            token_to_index[token] = index

        # Assign indices to each of the worker's vocab
        for worker, vocab in worker_to_vocab.items():

            # Map this vocab to indices
            indexed_vocab = self._map_to_indices(vocab, token_to_index)

            # Send the indexed vocab back to worker
            worker.receive_indexed_vocab(indexed_vocab)

    @staticmethod
    def _map_to_indices(vocab, token_to_index):
        """
        Args:
            vocab (set): Set of tokens
            token_to_index (dict): Maps each unique token to an index

        Returns:
            indexed_vocab (dict): Each token in vocab mapped to an index
        """
        indexed_vocab = dict()

        for token in vocab:
            index = token_to_index[token]
            indexed_vocab[token] = index

        return indexed_vocab

    @abstractmethod
    def _send_msg(self, message: bin, location: BaseWorker):
        raise NotImplementedError

    @abstractmethod
    def _recv_msg(self, message: bin) -> bin:
        raise NotImplementedError
