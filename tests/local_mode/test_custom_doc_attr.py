import syft as sy
import torch
import syfertext

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)

def test_add_custom_attr_doc():
    """Test adding custom attribute to Doc objects"""
    
    doc = nlp("Joey doesnt share food")
    doc.set_attribute(name = 'my_custom_tag', value = 'tag')
    
    # check custom tag has been added
    assert (hasattr(doc._,'my_custom_tag') and doc._.my_custom_tag == 'tag')