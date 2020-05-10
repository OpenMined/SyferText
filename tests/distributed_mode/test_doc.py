import syft as sy
import torch
import syfertext
from syft.generic.string import String
from syft.workers.virtual import VirtualWorker

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)

bob = VirtualWorker(hook, id="bob")

text_ptr = String("Private and Secure NLP").send(bob)

remote_doc = nlp(text_ptr)


def test_add_custom_attr():
    """Test add custom attribute to remote remote_doc."""

    # add new custom attributes
    remote_doc.set_attribute(name="doc_tag", value="doc_value")

    # check if the custom attributes exist
    assert remote_doc.has_attribute("doc_tag")


def test_remove_custom_attr():
    """Test removing a custom attribute from remote Doc"""

    # add new custom attributes
    remote_doc.set_attribute(name="doc_tag", value="doc_value")

    # check if the custom attributes exist
    assert remote_doc.has_attribute("doc_tag")

    # remove the new custom attributes
    remote_doc.remove_attribute(name="doc_tag")

    # Verify custom attribute was removed
    assert not remote_doc.has_attribute("doc_tag")


def test_get_custom_attr():
    """Test gett the value of custom attribute from remote Doc."""

    # add custom attributes
    remote_doc.set_attribute(name="doc_tag", value="doc_value")

    # Assert values are the values we set
    assert remote_doc.get_attribute("doc_tag") == "doc_value"


def test_update_custom_attr_doc():
    """Test updating custom attribute of Doc and Token object."""

    # add new custom attributes
    remote_doc.set_attribute(name="doc_tag", value="doc_value")

    # check custom attributes values are set
    assert remote_doc.get_attribute("doc_tag") == "doc_value"

    # now update the attributes
    remote_doc.set_attribute(name="doc_tag", value="new_doc_value")

    # now check the updated attribute
    assert remote_doc.get_attribute("doc_tag") == "new_doc_value"
