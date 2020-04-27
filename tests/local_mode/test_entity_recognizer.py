import syfertext
import syft as sy
import torch


from syfertext.pipeline import ner
from syfertext.doc import Doc

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)

entity = {"NAME": 0, "GPE": 1}


def test_entity_tagging():
    """Test entity recognizer tags the entities."""

    # Assuming a model based entity recognizer will give the following output
    # Input : (Albert Einstein was born in Germany)
    # Output: ((Albert, NAME), (Einstein, NAME), (was, ""), (born, ""), (in, ""), (Germany, "GPE")

    lookups = {"Albert": entity["NAME"], "Einstein": entity["NAME"], "Germany": entity["GPE"]}

    doc = nlp("Albert Einstein was born in Germany")

    defualt_tag = ""
    entity_recognizer = ner.EntityRecognizer(lookups=lookups, default_tag=defualt_tag)

    entity_recognizer(doc)  # modifies it in place

    # Check if tags were successfully stored in the right attributes
    ent_iob_tags = {
        "Albert": ner.begin_token_tag,
        "Einstein": ner.inner_token_tag,
        "was": ner.non_entity_token_tag,
        "born": ner.non_entity_token_tag,
        "in": ner.non_entity_token_tag,
        "Germany": ner.begin_token_tag,
    }

    ent_type_tags = {
        "Albert": entity["NAME"],
        "Einstein": entity["NAME"],
        "was": "",
        "born": "",
        "in": "",
        "Germany": entity["GPE"],
    }

    # Check if `ent_iob_` and `ent_type_` attributes of tokens are set to right value
    for token, ent_iob, ent_type in zip(doc, ent_iob_tags.values(), ent_type_tags.values()):

        assert token._.ent_iob_ == ent_iob
        assert token._.ent_type_ == ent_type
