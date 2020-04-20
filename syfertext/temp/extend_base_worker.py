# This functionality has be added to
# PySyft BaseWorker. I am in contact with them
# and will update on it soon.
# The tests pass on my local machine.

# Add these class variables
# self.private_key = None
# self.secret = None

# def generate_private_key(self, shared_prime):
#     """Generates a private key.
#     # TODO: Replace with a pseudo random number generator ??
#     Args:
#         shared_prime:
#     """
#
#     self.private_key = random.randint(1, shared_prime)
#
#
# def generate_public_key(self, shared_prime, shared_base):
#     if self.private_key is None:
#         self.generate_private_key(shared_prime)
#
#     public_key = (shared_base ** self.private_key) % shared_prime
#     return public_key
#
#
# def generate_secret_key(self, received_public_key, shared_prime):
#     assert (self.private_key is not None), "Please generate private key first."
#
#     secret = (received_public_key ** self.private_key) % shared_prime
#
#     # convert the secret key to bytes of lengths 16, 24 or 32
#     self.secret = secret.to_bytes(32, sys.byteorder)
