# This functionality can be added to can be added to PySyft BaseWorker.

from abc import abstractmethod
from syft.workers.base import BaseWorker

import random
import sys
from typing import Union, List


class ExtendedBaseWorker(BaseWorker):
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

        self.private_key = None
        self.secret = None

    def generate_private_key(self, shared_prime):
        """Generates a private key.
        # TODO: Replace with a pseudo random number generator ??
        Args:
            shared_prime:
        """

        self.private_key = random.randint(1, shared_prime)

    def generate_public_key(self, shared_prime, shared_base):
        """

        Args:
            shared_prime:
            shared_base:

        Returns:

        """
        if self.private_key is None:
            self.generate_private_key(shared_prime)

        public_key = (shared_base ** self.private_key) % shared_prime
        return public_key

    def generate_secret_key(self, received_public_key, shared_prime):
        """
        # TODO: Make secret and private key's private variables
        Args:
            received_public_key:
            shared_prime:

        Returns:

        """
        assert self.private_key is not None, "Please generate public key first."

        secret = (received_public_key ** self.private_key) % shared_prime

        # convert the secret key to bytes of lengths 16, 24 or 32
        self.secret = secret.to_bytes(32, sys.byteorder)

    @abstractmethod
    def _send_msg(self, message: bin, location: BaseWorker):
        raise NotImplementedError

    @abstractmethod
    def _recv_msg(self, message: bin) -> bin:
        raise NotImplementedError
