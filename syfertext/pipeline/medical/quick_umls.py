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
    # TODO : add accepted sem types option as well
    def __init__(
        self, 
        quick_umls_database = './umls_data_lower',
        threshold = 0.85, 
        overlapping_criteria = 'score',
        window = 5,
        similarity_name = 'jaccard', 
        keep_uppercase = False
    ):
        """ 
        Instantiate QuickUMLS object. This is the main interface through 
        which text can be processed.
        Args:
            quick_umls_database (str): Path to UMLS dictionary
            
            overlapping_criteria (str, optional):
                    One of "score" or "length". Choose how results are ranked.
                    Choose "score" for best matching score first or "length" for longest match first.. Defaults to 'score'.

            threshold (float, optional): Minimum similarity between strings. Defaults to 0.7.

            window (int, optional): Maximum amount of tokens to consider for matching. Defaults to 5.

            similarity_name (str, optional): One of "dice", "jaccard", "cosine", or "overlap".
                    Similarity measure to be used. Defaults to 'jaccard'.

            keep_uppercase (bool, optional): By default QuickUMLS converts all
                    uppercase strings to lowercase. This option disables that
                    functionality, which makes QuickUMLS useful for
                    distinguishing acronyms from normal words. 
                    
                    For this the database to be used must be `./umls_data`
                    Defaults to False.
        """
        pass


    def __call__(self, doc: Doc):
        pass