import syft as sy
import torch
from syft.generic.string import String
from syfertext.local_pipeline import get_test_language_model
from syfertext.utils import hash_string

hook = sy.TorchHook(torch)
me = hook.local_worker

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")


def test_pipeline_reuse():
    nlp = get_test_language_model()

    text1 = String('lazy dog jumps over sleepy cat')

    assert (len(nlp.subpipelines) == 0)
    assert ('me' not in nlp.pipeline)

    doc = nlp(text1)

    # only one sub-pipeline will be created
    assert (len(nlp.subpipelines) == 1)
    assert (len(nlp.pipeline['me']) == 1)

    subpipeline_hash1 = hash_string(string='me' + 'tokenizer')
    assert subpipeline_hash1 in nlp.subpipelines

    text2 = String('a cow says moo')
    doc = nlp(text2)

    # pipeline will be reused now, since text is on the me worker
    assert (len(nlp.subpipelines) == 1)
    assert (len(nlp.pipeline['me']) == 1)

    text3 = String('horse runs fast').send(bob)
    doc = nlp(text3)

    subpipeline_hash2 = hash_string(string='bob' + 'tokenizer')
    assert subpipeline_hash2 in nlp.subpipelines

    # a new sub-pipeline is created since text3 is on bob's machine
    assert (len(nlp.subpipelines) == 2)
    assert (len(nlp.pipeline['bob']) == 1)
