from syfertext.doc import Doc
from syfertext.span import Span
from syfertext.token import Token
from typing import Union

from syfertext.utils import Intervals, get_similarity

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
                    
                    For this, the database to be used must be `./umls_data`
                    Defaults to False.
        """
        pass


    def __call__(self, doc: Doc):
        # find all the matches

        matches = self._match(doc)

        # Now update the doc
        # where to put the matches in doc ??
        # make spans  ??
        # maybe in doc.ents ?? or in doc.medical_ents ??
        # definitely put in doc.cuis

        for match in matches:
            
            # TODO : maybe first add cuis in StringStore
            cuis = []
            for each_match in match:
                cuis.append(each_match['cui'])
            
            entity = Span(doc, match[0]['start'], match[0]['end'])

            # Set cuis in custom attr
            entity.set_attribute('cuis', cuis)
            entity.set_attribute('ent_type','medical term')

            doc.ents.append(entity)

            # to access definition use, quickUMLS.kb[cui] to extract definitions    

        return doc

    def _get_all_matches(self, ngrams):
        """Returns all matches in the form of list
        of dictionary of 
            match = {
                'start',
                'end',
                'ngram',
                'matched_text',
                'cui',
                'similarity'
            }
        """

        all_matches = []

        for start, end, ngram in ngrams:

            # Convert to lowercase if possible
            if not self.keep_uppercase:
                ngram = ngram.lower()
            
            # find all the candidates
            matches = list(searcher.get(ngram))

            for match in matches:

                ngram_matches = []
                
                # get all CUIs with this matched text
                # in database. Yes, one text can have more than
                # one CUI. for eg. 40 year old has CUI of age, adult etc. 
                cuis = self.string_to_cui[match]

                similarity = get_similarity(ngram, match, self.similarity_name)

                for cui in cuis:
                    if match_similarity == 0:
                        continue
                    
                    # TODO : Only include those which the user wants, eg, T053, T0151
                    # if not self._is_ok_semtype(semtypes):
                    #     continue

                    if prev_cui is not None and prev_cui == cui:
                        if match_similarity > ngram_matches[-1]['similarity']:
                            ngram_matches.pop(-1)
                        else:
                            continue

                    prev_cui = cui

                    ngram_matches.append(
                        {
                            'start': start,
                            'end': end,
                            'ngram': ngram,
                            'matched_text': match,
                            'cui': cui,
                            'similarity': match_similarity,
                            # 'semtypes': semtypes,
                            # 'preferred': preferred
                        }
                    )

            if len(ngram_matches) > 0:
                matches.append(
                    sorted(
                        ngram_matches,
                        key=lambda m: m['similarity'], # + m['preferred'],
                        reverse=True
                    )
                )
        
        return all_matches
    
    @staticmethod
    def _select_score(match):
        return (match[0]['similarity'], (match[0]['end'] - match[0]['start']))

    @staticmethod
    def _select_longest(match):
        return ((match[0]['end'] - match[0]['start']), match[0]['similarity'])

    def _select_terms(self, matches):
        sort_func = (
            self._select_longest if self.overlapping_criteria == 'length'
            else self._select_score
        )

        matches = sorted(matches, key=sort_func, reverse=True)

        intervals = Intervals()
        final_matches_subset = []

        for match in matches:
            match_interval = (match[0]['start'], match[0]['end'])
            if match_interval not in intervals:
                final_matches_subset.append(match)
                intervals.append(match_interval)

        return final_matches_subset 

    # NOTE : we are not currently using the heuristcs introduced in the paper (Soldaini and Goharian, 2016).
    # Also make better doc strings
    def _match(self, doc : Doc, best_match=True):
        """Perform UMLS concept resolution for the given string.
        Args:
            text (str): Text on which to run the algorithm
            best_match (bool, optional): Whether to return only the top match or all overlapping candidates. Defaults to True.

        Returns:
            List: List of all matches in the text
        """

        ngrams = self._make_ngrams(doc)

        matches = self._get_all_matches(ngrams)

        if best_match:
            matches = self._select_terms(matches)

        return matches
