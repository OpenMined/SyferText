import syft as sy
from syft.generic.string import String
import syfertext
from syfertext.local_pipeline import get_test_language_model
from syfertext.pipeline.single_label_classifier import AverageDocEncoder, SingleLabelClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from syfertext.utils import hash_string

hook = sy.TorchHook(torch)
me = hook.local_worker
me.is_client_worker = False

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
james = sy.VirtualWorker(hook, id="james")
dan = sy.VirtualWorker(hook, id="dan")
bill = sy.VirtualWorker(hook, id="bill")


class simpleNet(sy.Plan):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        logits = self.fc(x)
        probs = F.relu(logits)
        return probs, logits


doc_encoder = AverageDocEncoder()
# Create some temporary labels
labels = ["A", "B"]


def reset_object_store(worker):
    keys = list(worker._objects)
    for k in keys:
        del worker._objects[k]


def test_pipeline_reuse():

    text1 = String("This string is on alice").send(alice)
    text2 = String("This string is on james").send(james)

    nlp = get_test_language_model()
    pipe_name = nlp.pipeline_name

    # Set up SingleLabelClassifier and add to pipeline
    model = simpleNet()
    classifier = SingleLabelClassifier(
        classifier=model, doc_encoder=doc_encoder, encryption="mpc", labels=labels,
    )

    nlp.add_pipe(classifier, name="test_classifier", access={"bob"})
    nlp.deploy(bob)

    # Clear my object store
    reset_object_store(me)

    deployed_nlp = syfertext.load(pipe_name)

    doc1 = deployed_nlp(text1)

    # Now 2 subpipelines were created, one for Tokenizer at Alice (Subpipeline[Tokenizer]), and one for Classfier at Bob(Subpipeline[SingleLabelClassifier])
    assert len(deployed_nlp.subpipelines) == 2

    subpipeline_hash1 = hash_string(string="alice" + "tokenizer")
    assert subpipeline_hash1 in deployed_nlp.subpipelines

    subpipeline_hash2 = hash_string(string="bob" + "test_classifier")
    assert subpipeline_hash2 in deployed_nlp.subpipelines

    doc2 = deployed_nlp(text2)

    # now only one extra subpipeline was created at James (Subpipeline[Tokenizer])
    # Classfier at Bob(Subpipeline[SingleLabelClassifier]) was re-used.
    assert len(deployed_nlp.subpipelines) == 3

    subpipeline_hash3 = hash_string(string="james" + "tokenizer")
    assert subpipeline_hash3 in deployed_nlp.subpipelines
