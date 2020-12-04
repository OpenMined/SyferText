import syft as sy
import torch
import syfertext
from syfertext.pipeline import SubPipeline, SimpleTagger

# Import PySyft's String class
from syft.generic.string import String
from syfertext.local_pipeline import get_test_language_model
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

import os
from pathlib import Path

hook = sy.TorchHook(torch)
# The local worker
me = hook.local_worker
me.is_client_worker = False


def reset_object_store(worker):
    keys = list(worker._objects)
    for k in keys:
        del worker._objects[k]


bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# The remote workers
# bob = DataCentricFLClient(hook, "http://18.220.216.78:5001/")
# alice = DataCentricFLClient(hook, "http://18.220.216.78:5002/")

# The crypto provider
# crypto_provider = DataCentricFLClient(hook, "http://18.220.216.78:5003/")
# my_grid = sy.PrivateGridNetwork(bob, alice, crypto_provider, me)

nlp = get_test_language_model()
# type(nlp), nlp.owner

# tagger = SimpleTagger(attribute="noun", lookups=["SyferText"], tag=True)
# nlp.add_pipe(tagger, name="noun_tagger",access = {me})


print(nlp.pipeline_template)
print(nlp.states_info)
nlp.deploy(bob)
print(nlp.states_info)

print(me._objects)

reset_object_store(me)

print(me._objects)

cache_directory = os.path.join(str(Path.home()), "SyferText", "cache", "syfertext_sentiment")

if os.path.exists(cache_directory):
        for f in os.listdir(cache_directory):
            os.remove(cache_directory + "/" + f)

        os.rmdir(cache_directory)

# bob.connect()
# Loading the deployed model
nlp_deployed = syfertext.load("syfertext_sentiment")

nlp_load = syfertext.load("syfertext_sentiment")


if nlp_deployed == nlp_load:
    print("Both are same")
else:
    print("Different objects")

# Testing nlp_deployed
doc = nlp_deployed("This is some random test")
print(nlp_deployed.pipeline_template)
print(nlp_deployed.states_info)
print(nlp_deployed.deployed_on)

print(nlp_load.pipeline_template)
print(nlp_load.states_info)
print(nlp_load.deployed_on)
for token in doc:
    print(token)
