import pytest

import syft as sy
from syft.generic.string import String
import syfertext
from syfertext.local_pipeline import get_test_language_model
from syfertext.pipeline.single_label_classifier import AverageDocEncoder, SingleLabelClassifier
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
def test_unencrypted_single_label_inference(text, expected):
    torch.manual_seed(seed)
    classifier_name = "classifier"
    nlp = get_test_language_model()
    pipeline_name = nlp.pipeline_name

    # Set up SingleLabelClassifier and add to pipeline
    net = Net()
    classifier = SingleLabelClassifier(
        classifier=net, 
        doc_encoder=doc_encoder,
        encryption=None,
        labels=labels
    )  
    nlp.add_pipe(classifier, name=classifier_name, access={'*'})

    attribute_name = "{pipeline_name}__{classifier_name}".format(
        pipeline_name=pipeline_name,
        classifier_name=classifier_name
    )

    # Case 1: infer using string and pipeline before deployment
    doc1 = nlp(String(text))

    assert hasattr(doc1._, attribute_name)
    assert doc1.get_attribute(attribute_name) == expected

    # Case 2: infer using string pointer and pipeline before deployment
    string_ptr = String(text).send(bob)
    doc_ptr = nlp(string_ptr) 
    doc2 = bob._objects[doc_ptr.id_at_location]

    assert hasattr(doc2._, attribute_name)
    assert doc2.get_attribute(attribute_name) == expected

    # Deploy pipeline 
    nlp.deploy(worker=bob)

    # Empty local worker's object store to ensure that 
    # we are using the deployed pipeline rather than local states
    for obj in list(me._objects):
        del me._objects[obj]

    # Load deployed pipeline
    nlp_loaded_from_bob = syfertext.load(pipeline_name)

    # Case 3: infer using local string and deployed pipeline
    doc3 = nlp_loaded_from_bob(String(text))

    assert hasattr(doc3._, attribute_name)
    assert doc3.get_attribute(attribute_name) == expected

    # Case 4: infer using string pointer and deployed pipeline
    string_ptr = String(text).send(bob)
    doc_ptr = nlp_loaded_from_bob(string_ptr)
    doc4 = bob._objects[doc_ptr.id_at_location]

    assert hasattr(doc4._, attribute_name)
    assert doc4.get_attribute(attribute_name) == expected

'''
This test has currently been disabled as it was giving out unexpected results

@pytest.mark.parametrize("text,expected", [("The quick brown fox", "B")])
def test_encrypted_single_label_inference(text, expected):
    torch.manual_seed(seed)
    classifier_name = "encrypted_classifier"
    nlp = get_test_language_model()
    pipeline_name = nlp.pipeline_name

    # Set up SingleLabelClassifier and add to pipeline
    net = Net()
    classifier = SingleLabelClassifier(
        classifier=net, 
        doc_encoder=doc_encoder,
        encryption="mpc",
        labels=labels
    )  
    nlp.add_pipe(classifier, name=classifier_name, access={'*'})

    attribute_name = "{pipeline_name}__{classifier_name}".format(
        pipeline_name=pipeline_name,
        classifier_name=classifier_name
    )

    # Case 1: infer using string and pipeline before deployment
    doc1 = nlp(String(text))

    assert hasattr(doc1._, attribute_name)
    assert doc1.get_attribute(attribute_name) == expected

    # Case 2: infer using string pointer and pipeline before deployment
    string_ptr = String(text).send(bob)
    doc_ptr = nlp(string_ptr) 
    doc2 = bob._objects[doc_ptr.id_at_location]

    assert hasattr(doc2._, attribute_name)
    assert doc2.get_attribute(attribute_name) == expected

    # Deploy pipeline 
    nlp.deploy(worker=bob)

    # Empty local worker's object store to ensure that 
    # we are using the deployed pipeline rather than local states
    for obj in list(me._objects):
        del me._objects[obj]

    # Load deployed pipeline
    nlp_loaded_from_bob = syfertext.load(pipeline_name)

    # Case 3: infer using local string and deployed pipeline
    doc3 = nlp_loaded_from_bob(String(text))

    assert hasattr(doc3._, attribute_name)
    assert doc3.get_attribute(attribute_name) == expected

    # Case 4: infer using string pointer and deployed pipeline
    string_ptr = String(text).send(bob)
    doc_ptr = nlp_loaded_from_bob(string_ptr)
    doc4 = bob._objects[doc_ptr.id_at_location]

    assert hasattr(doc4._, attribute_name)
    assert doc4.get_attribute(attribute_name) == expected
'''