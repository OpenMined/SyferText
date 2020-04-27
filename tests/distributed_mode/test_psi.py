import syfertext
import syft as sy
import sys
import torch

from syft.generic.string import String
from syfertext.workers.virtual import VirtualWorker

from syfertext.encdec import encrypt, decrypt

hook = sy.TorchHook(torch)
me = sy.local_worker

# create workers
bob = VirtualWorker(hook, "bob")
alice = VirtualWorker(hook, "alice")
alan = VirtualWorker(hook, "alan")
neo = VirtualWorker(hook, "neo")

nlp = syfertext.load("en_core_web_lg", owner=me)

shared_prime = 997
shared_base = 2
key = 1009
temp_key = key.to_bytes(32, sys.byteorder)


def test_deterministic_encryption():
    """Test that encryption of same string with the same key
    returns the same cipher."""

    text = "SyferText"

    # Encrypt text twice
    enc_text = encrypt(text, temp_key)
    enc_text_again = encrypt(text, temp_key)

    assert enc_text == enc_text_again


def test_encrypt_decrypt():
    """Test the encrypt and decrypt functions."""

    text = "SyferText"

    # Encrypt text
    enc_text = encrypt(text, temp_key)

    # Decrypt text
    dec_text = decrypt(enc_text, temp_key).decode("utf-8")

    # assert decrypted text is same as original text
    assert text == dec_text


def test_two_party_diffie_hellman_key_exchange():
    """Test the Diffie-Hellman Key Exchange protocol.
    By encrypting same string we will verify that at the end of
    the exchange bob and alice have the same keys.
    """

    # Send same strings to bob and alice
    text = String("syfertext")

    bob_doc = nlp(text.send(bob))
    alice_doc = nlp(text.send(alice))

    # Let james be our good guy here
    secure_worker = VirtualWorker(hook, id="james")

    workers = [bob, alice]

    # Execute the DH key exchange protocol securely on Secure Worker
    secure_worker.execute_dh_key_exchange(shared_prime, shared_base, workers)

    bob_enc_tokens = bob_doc.get_encrypted_tokens_set()
    alice_enc_tokens = alice_doc.get_encrypted_tokens_set()

    # assert the sets are same, thus verifying the keys are same
    assert not bob_enc_tokens.difference(alice_enc_tokens)


def test_multi_party_diffie_hellman_key_exchange():
    """Test the Diffie-Hellman Key Exchange protocol for more than 2 parties.
    By encrypting same string we will verify that at the end of
    the exchange bob and alice have the same keys.
    """

    text = String("syfertext")

    # List of workers participating in DH Key exchange protocol
    workers = [bob, alice, alan, neo]

    docs = list()

    for worker in workers:

        cur_worker_doc = nlp(text.send(worker))

        docs.append(cur_worker_doc)

    # Let james again be our good guy here
    secure_worker = VirtualWorker(hook, id="james")

    # Execute the DH key exchange protocol securely on Jame's (SecureWorker) machine
    secure_worker.execute_dh_key_exchange(shared_prime, shared_base, workers)

    # List of sets of encrypted tokens returned from each workers's doc
    enc_token_sets = list()

    for doc in docs:

        cur_worker_set = doc.get_encrypted_tokens_set()

        enc_token_sets.append(cur_worker_set)

    # assert the sets are same, thus verifying the keys are same
    for set1 in enc_token_sets:

        for set2 in enc_token_sets:

            if set1 is set2:  # both point to the same object
                continue

            # assert sets are same
            assert not set1.difference(set2)


def test_two_party_psi():
    """Test two party private set intersection."""

    # NOTE: Keys were generated previous tests

    # Simulate private dataset
    bob_private_data = String("private and secure nlp").send(bob)
    bob_doc = nlp(bob_private_data)

    alice_private_data = String("keeps your data private and secure").send(alice)
    alice_doc = nlp(alice_private_data)

    bob_enc_tokens = bob_doc.get_encrypted_tokens_set()
    alice_enc_tokens = alice_doc.get_encrypted_tokens_set()

    unique_tokens = 7  # [private, and, secure, nlp, keeps, your,data]
    common_tokens = 3  # [private, and, secure]

    # Verify length of union is same as number of unique_tokens
    assert len(bob_enc_tokens.union(alice_enc_tokens)) == unique_tokens

    # Verify length of intersection of sets is same as number of common tokens
    assert len(bob_enc_tokens.intersection(alice_enc_tokens)) == common_tokens

    # Assign indices to all tokens
    # index = 0
    # token_to_index = {}
    # for token in enc_tokens:
    #     # map token to index
    #     token_to_index[token] = index
    #     # increment index
    #     index += 1
    #
    # # map bob's tokens to indices
    # bob_token_to_index = {}
    #
    # for token in bob_enc_tokens:
    #     # map token to index
    #     bob_token_to_index[token] = token_to_index[token]
    #
    # # map alice's tokens to indices
    # alice_token_index = {}
    #
    # for token in alice_enc_tokens:
    #     # map token to index
    #     alice_token_index[token] = token_to_index[token]
