import pytest

import syft as sy
from syft.generic.string import String
from syfertext.pipeline.single_label_classifier import AverageDocEncoder, SingleLabelClassifier
from syfertext.local_pipeline import get_test_language_model
import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a torch hook for PySyft
hook = sy.TorchHook(torch)
seed = 42

# Set up local and remote workers
me = hook.local_worker
me.is_client_worker = False

alice = sy.VirtualWorker(hook, id='alice')
bob = sy.VirtualWorker(hook, id='bob')
crypto_provider = sy.VirtualWorker(hook, id='crypto_provider')

# Create toy classifier
class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        logits = self.fc(x)
        probs = F.relu(logits)
        return probs, logits

doc_encoder = AverageDocEncoder()
labels = ["A", "B"]



@pytest.mark.parametrize("text,expected", [("The quick brown fox", "B")])
def test_single_label_inference(text, expected):
    torch.manual_seed(seed)
    classifier_name = "classifier"
    nlp = get_test_language_model()

    # Set up SingleLabelClassifier and add to pipeline
    net = Net()
    classifier = SingleLabelClassifier(
        classifier=net, 
        doc_encoder=doc_encoder,
        encryption=None,
        labels=labels
    )  
    nlp.add_pipe(classifier, name=classifier_name, access={'*'})
    nlp.deploy(worker=bob)

    string = String(text)
    doc_ptr = nlp(string) 

    # Check if actual label matches expected label
    attribute_name = "{pipeline_name}__{classifier_name}".format(
        pipeline_name=nlp.pipeline_name,
        classifier_name=classifier_name
        )
    assert hasattr(doc_ptr._, attribute_name)
    assert doc_ptr.get_attribute(attribute_name) == expected


@pytest.mark.parametrize("text,expected", [("The quick brown fox", "B")])
def test_encrypted_single_label_inference(text, expected):
    torch.manual_seed(seed)
    classifier_name = "encrypted_classifier"
    nlp = get_test_language_model()

    # Set up SingleLabelClassifier and add to pipeline
    net = Net()
    classifier = SingleLabelClassifier(
        classifier=net, 
        doc_encoder=doc_encoder,
        encryption="mpc",
        labels=labels
    )  
    nlp.add_pipe(classifier, name=classifier_name, access={'*'})
    nlp.deploy(worker=bob)

    string = String(text)
    doc_ptr = nlp(string) 

    # Check if actual label matches expected label
    attribute_name = "{pipeline_name}__{classifier_name}".format(
        pipeline_name=nlp.pipeline_name,
        classifier_name=classifier_name
        )
    assert hasattr(doc_ptr._, attribute_name)
    assert doc_ptr.get_attribute(attribute_name) == expected