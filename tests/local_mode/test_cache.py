import syft as sy
import torch
import syfertext
import os
from pathlib import Path

# Import PySyft's String class
from syft.generic.string import String
from syfertext.local_pipeline import get_test_language_model

hook = sy.TorchHook(torch)
me = hook.local_worker
me.is_client_worker = False


def reset_object_store(worker):
    """ This functions resets all the objects for the specified worker """
    keys = list(worker._objects)
    for k in keys:
        del worker._objects[k]


# Remote Virutal Workers
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

nlp = get_test_language_model()
nlp.deploy(bob)

# Resetting the object store
reset_object_store(me)


def test_save_to_storage():
    nlp_deployed = syfertext.load("syfertext_sentiment")

    cache_directory = os.path.join(str(Path.home()), "SyferText", "cache", "syfertext_sentiment")

    assert os.path.exists(cache_directory)
