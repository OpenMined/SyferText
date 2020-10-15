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


# The remote workers
bob = DataCentricFLClient(hook, "http://18.220.216.78:5001/")
alice = DataCentricFLClient(hook, "http://18.220.216.78:5002/")

# The crypto provider
crypto_provider = DataCentricFLClient(hook, "http://18.220.216.78:5003/")
my_grid = sy.PrivateGridNetwork(bob, alice, crypto_provider, me)

nlp = get_test_language_model()
type(nlp), nlp.owner

tagger = SimpleTagger(attribute="noun", lookups=["SyferText"], tag=True)
# nlp.add_pipe(tagger, name="noun_tagger",access = {me})
nlp.deploy(bob)

# Loading the deployed model
nlp_deployed = syfertext.load("syfertext_sentiment")

# Testing nlp_deployed
doc = nlp_deployed("This is some random test")
print(nlp_deployed.pipeline_template)
for token in doc:
    print(token)
