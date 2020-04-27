# This functionality can be added to can be added to PySyft BaseWorker.

from abc import abstractmethod
from syft.workers.base import BaseWorker

import random
import sys
from typing import Union, List


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

    def generate_private_key(self, shared_prime):
        """Generates a private key.
        """

        # TODO: Replace with a pseudo random number generator ??
        self.private_key = random.randint(1, shared_prime)

    def generate_public_key(self, shared_prime, shared_base):
        """ Generates and returns a public key.
        """

        if self.private_key is None:
            self.generate_private_key(shared_prime)

        public_key = (shared_base ** self.private_key) % shared_prime
        return public_key

    def generate_secret_key(self, shared_prime, received_public_key):
        """ Generates a secret key for the worker using the received_public_key.
        """

        if self.private_key is None:
            self.generate_private_key(shared_prime)

        self.secret = (received_public_key ** self.private_key) % shared_prime

        # Convert the secret key to bytes of lengths 16, 24 or 32
        # Useful for AES encryption
        self.secret_key = self.secret.to_bytes(32, sys.byteorder)

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

    @abstractmethod
    def _send_msg(self, message: bin, location: BaseWorker):
        raise NotImplementedError

    @abstractmethod
    def _recv_msg(self, message: bin) -> bin:
        raise NotImplementedError
