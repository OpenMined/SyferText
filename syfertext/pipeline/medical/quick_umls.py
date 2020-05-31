from syfertext.doc import Doc
from syfertext.token import Token
from typing import Union

from quickumls_simstring import simstring

class QuickUMLS:
    """ Extracts medical entities using
    approximation matching. Also does UMLS linking by
    returning respective CUIs of matched entities.

    Inspired from :
        QuickUMLS : https://github.com/Georgetown-IR-Lab/QuickUMLS
        For more info, refer to the paper : http://medir2016.imag.fr/data/MEDIR_2016_paper_16.pdf
    """

    def __init__(self):
        pass

    def __call__(self, doc: Doc):
        pass

    