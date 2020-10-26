import syft as sy
import torch
import syfertext
from syfertext.pipeline import SubPipeline, SimpleTagger

# Import PySyft's String class
from syft.generic.string import String
from syfertext.local_pipeline import get_test_language_model
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

hook = sy.TorchHook(torch)
# The local worker
me = hook.local_worker
me.is_client_worker = False


def reset_object_store(worker):
    keys = list(worker._objects)
    for k in keys:
        del worker._objects[k]

bob = sy.VirtualWorker(hook, id = 'bob')
alice = sy.VirtualWorker(hook, id = 'alice')

# The remote workers
# bob = DataCentricFLClient(hook, "http://18.220.216.78:5001/")
# alice = DataCentricFLClient(hook, "http://18.220.216.78:5002/")

# The crypto provider
# crypto_provider = DataCentricFLClient(hook, "http://18.220.216.78:5003/")
# my_grid = sy.PrivateGridNetwork(bob, alice, crypto_provider, me)

nlp = get_test_language_model()
type(nlp), nlp.owner

tagger = SimpleTagger(attribute="noun", lookups=["SyferText"], tag=True)
# nlp.add_pipe(tagger, name="noun_tagger",access = {me})
nlp.deploy(bob)

print(me._objects)

print(me._objects['syfertext_sentiment:tokenizer'])

reset_object_store(me)

print(me._objects)

# bob.connect()
# Loading the deployed model
nlp_deployed = syfertext.load("syfertext_sentiment")

# Testing nlp_deployed
doc = nlp_deployed("This is some random test")
print(nlp_deployed.pipeline_template)
for token in doc:
    print(token)
