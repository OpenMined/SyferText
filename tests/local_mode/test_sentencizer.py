import syft
from syft.generic.string import String
import torch
import syfertext
from syfertext.pipeline import Sentencizer
from syfertext.pointers import SpanPointer

# Create a torch hook for PySyft
hook = syft.TorchHook(torch)

# Get a reference to the local worker
me = hook.local_worker

# Define a text to divide into sentences
text = "The quiCk broWn Fox jUmps over thE lazY Dog. I will tokenizE thiS phrase wiTh SyferText! I Will do it myselF ??"
gt_sentences = [
    "The quiCk broWn Fox jUmps over thE lazY Dog. ",
    "I will tokenizE thiS phrase wiTh SyferText! ",
    "I Will do it myselF ??",
]


def test_sentencizer_local():
    """Test the sentencizer pipeline on a local text"""

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)

    # instantiate a setencizer and add it to the pipeline
    sent = Sentencizer()
    nlp.add_pipe(sent, name="sentencizer")

    # pass the text through the pipeline
    doc = nlp(text)

    all_sentences = [sent for sent in doc.sents]

    assert len(all_sentences) == 3

    for i, sentence in enumerate(all_sentences):
        assert sentence.text == gt_sentences[i]


def test_sentencizer_remote():
    """Test the sentencizer pipeline on a local text"""

    bob = syft.VirtualWorker(hook, id="bob")
    remote_text = String(text).send(bob)

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)

    # instantiate a setencizer and add it to the pipeline
    sent = Sentencizer()
    nlp.add_pipe(sent, name="sentencizer", remote=True)

    # pass the text through the pipeline
    doc = nlp(remote_text)

    local_doc = nlp(text)

    all_sentences = [sent for sent in doc.sents]

    all_sentences_local = [sent for sent in local_doc.sents]

    assert len(all_sentences) == 3

    for i, sentence in enumerate(all_sentences):
        assert isinstance(sentence, SpanPointer)
        assert len(sentence) == len(all_sentences_local[i])
        assert sentence.location == bob
