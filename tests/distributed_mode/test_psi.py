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


def test_diffie_hellman_key_exchange():
    """Test the Diffie-Hellman Key Exchange protocol for more than 2 parties."""

    text = String("nlp")

    # List of workers participating in DH Key exchange protocol
    workers = [david, carol, alan, neo]

    docs = list()

    # Simulate private dataset across multiple workers
    for worker in workers:
        text_ptr = text.send(worker)
        doc = nlp(text_ptr)
        docs.append(doc)

    # Let james again be our good guy here
    secure_worker = VirtualWorker(hook, id="secure_worker")

    # Execute the DH key exchange protocol securely on Jame's (SecureWorker) machine
    secure_worker.execute_dh_key_exchange(shared_prime, shared_base, workers)

    # TODO: How to verify now ?


def test_combining_vocabulary():

    workers = [david, carol]
    secure_worker = VirtualWorker(hook, id="secure_worker")

    # Execute the DH key exchange protocol securely on Secure Worker
    secure_worker.execute_dh_key_exchange(shared_prime, shared_base, workers)

    # Add stop word tagger to the pipeline
    stop_word_tagger = SimpleTagger(attribute="is_stop", lookups=["and", "the", "are"], tag=True)
    nlp.add_pipe(stop_word_tagger, name="stop tagger", remote=True)

    # Simulate private dataset
    david_private_data = String("the quick brown fox jumps over the lazy dog").send(david)
    david_doc = nlp(david_private_data)

    carol_private_data = String("the dog and fox are good friends").send(carol)
    carol_doc = nlp(carol_private_data)

    dataset = [{"data": david_doc}, {"data": carol_doc}]
    excluded_tokens = {"is_stop": {True}}

    vocab_size = secure_worker.create_vocabulary(dataset, "data", excluded_tokens=excluded_tokens)

    assert vocab_size == len(
        ["quick", "brown", "fox", "jumps", "over", "lazy", "dog", "good", "freinds"]
    )
