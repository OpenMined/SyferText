import syfertext
import syft as sy
import sys
import torch

import random
from syft.generic.string import String
from syfertext.workers.virtual import VirtualWorker
from syfertext.pipeline import SimpleTagger

from syfertext.encdec import encrypt, decrypt
from syfertext.encdec import shared_prime, shared_base

hook = sy.TorchHook(torch)
me = VirtualWorker(hook, "local_worker")

nlp = syfertext.load("en_core_web_lg", owner=me)

# create workers
david = VirtualWorker(hook, "david")
carol = VirtualWorker(hook, "carol")
alan = VirtualWorker(hook, "alan")
neo = VirtualWorker(hook, "neo")

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
    the exchange david and carol have the same keys.
    """

    # Send same strings to david and carol
    text = String("syfertext")

    david_doc = nlp(text.send(david))
    carol_doc = nlp(text.send(carol))

    # Let james be our good guy here
    secure_worker = VirtualWorker(hook, id="secure_worker")

    workers = [david, carol]

    # Execute the DH key exchange protocol securely on Secure Worker
    secure_worker.execute_dh_key_exchange(shared_prime, shared_base, workers)

    david_enc_tokens = david_doc.get_encrypted_tokens_set()
    carol_enc_tokens = carol_doc.get_encrypted_tokens_set()

    # assert the sets are same, thus verifying the keys are same
    assert not david_enc_tokens.difference(carol_enc_tokens)


def test_multi_party_diffie_hellman_key_exchange():
    """Test the Diffie-Hellman Key Exchange protocol for more than 2 parties.
    By encrypting same string we will verify that at the end of
    the exchange david and carol have the same keys.
    """

    text = String("nlp")

    # List of workers participating in DH Key exchange protocol
    workers = [david, carol, alan, neo]

    docs = list()

    for worker in workers:

        text_ptr = text.send(worker)

        cur_worker_doc = nlp(text_ptr)

        docs.append(cur_worker_doc)

    # Let james again be our good guy here
    secure_worker = VirtualWorker(hook, id="secure_worker")

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


def test_get_encrypted_tokens_set():
    """Test get encrypted tokens set from workers."""

    workers = [david, carol]
    secure_worker = VirtualWorker(hook, id="secure_worker")

    # Execute the DH key exchange protocol securely on Secure Worker
    secure_worker.execute_dh_key_exchange(shared_prime, shared_base, workers)

    # Add stop word tagger to the pipeline
    stop_word_tagger = SimpleTagger(attribute="is_stop", lookups=["and", "your"], tag=True)
    nlp.add_pipe(stop_word_tagger, name="stop tagger", remote=True)

    # Simulate private dataset
    david_private_data = String("private and secure nlp").send(david)
    david_doc = nlp(david_private_data)

    carol_private_data = String("keeps your data private and secure").send(carol)
    carol_doc = nlp(carol_private_data)

    david_enc_tokens = david_doc.get_encrypted_tokens_set()
    carol_enc_tokens = carol_doc.get_encrypted_tokens_set()

    unique_tokens = 7  # [private, and, secure, nlp, keeps, your,data]
    common_tokens = 3  # [private, and, secure]

    # Verify length of union is same as number of unique_tokens
    assert len(david_enc_tokens.union(carol_enc_tokens)) == unique_tokens

    # Verify length of intersection of sets is same as number of common tokens
    assert len(david_enc_tokens.intersection(carol_enc_tokens)) == common_tokens

    # Get set excluding stop words
    david_enc_tokens = david_doc.get_encrypted_tokens_set(excluded_tokens={"is_stop": {True}})
    carol_enc_tokens = carol_doc.get_encrypted_tokens_set(excluded_tokens={"is_stop": {True}})

    unique_tokens = 5  # [private, secure, nlp, keeps, data]
    common_tokens = 2  # [private, secure]

    # Verify length of union is same as number of unique_tokens
    assert len(david_enc_tokens.union(carol_enc_tokens)) == unique_tokens

    # Verify length of intersection of sets is same as number of common tokens
    assert len(david_enc_tokens.intersection(carol_enc_tokens)) == common_tokens


def test_set_indices_and_get_indices():
    """Test Doc returns a list of indices representing its tokens."""

    workers = [david, carol]
    secure_worker = VirtualWorker(hook, id="secure_worker")

    # Execute the DH key exchange protocol securely on Secure Worker
    secure_worker.execute_dh_key_exchange(shared_prime, shared_base, workers)

    # Add stop word tagger to the pipeline
    stop_word_tagger = SimpleTagger(attribute="is_stop", lookups=["and", "your"], tag=True)
    nlp.add_pipe(stop_word_tagger, name="stop tagger", remote=True)

    # Simulate private dataset
    david_private_data = String("private and secure nlp").send(david)
    david_doc = nlp(david_private_data)

    carol_private_data = String("keeps your data private and secure").send(carol)
    carol_doc = nlp(carol_private_data)

    david_enc_tokens = david_doc.get_encrypted_tokens_set(excluded_tokens={"is_stop": {True}})
    carol_enc_tokens = carol_doc.get_encrypted_tokens_set(excluded_tokens={"is_stop": {True}})

    # Take a union of both sets
    all_tokens = david_enc_tokens.union(carol_enc_tokens)

    token_to_index = dict()

    # Assign each unique token to an index
    for i, enc_token in enumerate(all_tokens):
        token_to_index[enc_token] = i

    def map_to_indices(enc_tokens, token_to_index):
        """
        Args:
            enc_tokens (set): Set of encrypted tokens
            token_to_index (dict): Maps each unique token to an index

        Returns:
            mapped_enc_tokens (dict): Each token in enc_tokens
                mapped to an index
        """
        mapped_enc_tokens = dict()

        for token in enc_tokens:
            index = token_to_index[token]
            mapped_enc_tokens[token] = index

        return mapped_enc_tokens

    david_mapped_tokens = map_to_indices(david_enc_tokens, token_to_index)
    carol_mapped_tokens = map_to_indices(carol_enc_tokens, token_to_index)

    david_doc.set_indices(david_mapped_tokens)
    carol_doc.set_indices(carol_mapped_tokens)

    # Get tuple of indices
    david_tokens = david_doc.get_indices()
    carol_tokens = carol_doc.get_indices()

    assert david_tokens.shape[0] == 3  # [private, secure, nlp]
    assert carol_tokens.shape[0] == 4  # [keeps, data, private, secure]

    count = 0
    for i1 in david_tokens:
        for i2 in carol_tokens:
            if i1 == i2:
                count += 1

    assert count == 2  # [private, secure]
