import torch

import syft as sy

import syfertext
from syfertext.pipeline import SimpleTagger
from syfertext.local_pipeline import get_test_language_model
from syfertext.tokenizer import Tokenizer
from syfertext.vocab import Vocab
from syft.generic.string import String
from syfertext import utils

# Create a torch hook for PySyft
hook = sy.TorchHook(torch)

# Create a PySyft workers
me = hook.local_worker  # <- This is the worker from which we manage the processing
me.is_client_worker = False

# Initialize a farly extensive list of stop words from https://meta.wikimedia.org/wiki/Stop_word_list/google_stop_word_list#English

# Initialize an nlp pipeline that by default contains a tokenizer.
nlp = get_test_language_model()


text = String("thereafter a various ")
# apply in sequence tokenizer->stop_tagger->article_tagger
doc = nlp(text)
