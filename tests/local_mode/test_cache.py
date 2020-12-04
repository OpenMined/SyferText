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

    cache_directory = os.path.join(str(Path.home()), "SyferText", "cache", "syfertext_sentiment")

    # Remove any existing folder
    if os.path.exists(cache_directory):
        for f in os.listdir(cache_directory):
            os.remove(cache_directory + "/" + f)

        os.rmdir(cache_directory)

    nlp_to_save = syfertext.load("syfertext_sentiment")

    # Assert that the directory now exists
    assert os.path.exists(cache_directory)

    # Given that the directory now exists load it again from cache

    nlp_to_load = syfertext.load("syfertext_sentiment")

    assert nlp_to_save.pipeline_template == nlp_to_load.pipeline_template
    assert nlp_to_save.states_info == nlp_to_load.states_info
