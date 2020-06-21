from syfertext.doc import Doc
from syfertext.span import Span
from syfertext.token import Token
from typing import Union

from syfertext.utils import Intervals, get_similarity, SimstringDBReader
import pickle

from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde

import nltk

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
        quick_umls_database = '/home/jatinprakash/Desktop/notebooks/database_umls/umls-terms.simstring', # Change later
        knowledge_base = '/home/jatinprakash/Desktop/notebooks/kb_umls.pt', # Change later
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

        valid_criteria = {'length', 'score'}
        err_msg = (
            '"{}" is not a valid overlapping_criteria. Choose '
            'between {}'.format(
                overlapping_criteria, ', '.join(valid_criteria)
            )
        )
        assert overlapping_criteria in valid_criteria, err_msg
        self.overlapping_criteria = overlapping_criteria

        valid_similarities = {'dice', 'jaccard', 'cosine', 'overlap'}
        err_msg = ('"{}" is not a valid similarity name. Choose between '
                   '{}'.format(similarity_name, ', '.join(valid_similarities)))
        assert not(valid_similarities in valid_similarities), err_msg
        self.similarity_name = similarity_name

       
        # CUI Types.

        self.window = window
        self.ngram_length = 3
        self.threshold = threshold
        self.keep_uppercase = keep_uppercase # TODO Change this later, also make database of lowercase tokens

        self.quick_umls_database = quick_umls_database
        self.knowledge_base = knowledge_base

        # Initialize Simstring reader
        self.searcher = None
        self.string_to_cui = None

        self.min_match_length = 3

        self._stopwords = list(nltk.corpus.stopwords.words('english'))
        self.negations = ['none', 'non', 'neither', 'nor', 'no', 'not']
    
    def factory(self):
        """Creates a clone of this object.
        This method is used by the SupPipeline class to create
        objects using subpipeline templates.
        """

        return QuickUMLS(
            quick_umls_database = self.quick_umls_database, # Change later
            knowledge_base = self.knowledge_base, # Change later
            threshold = self.threshold, 
            overlapping_criteria = self.overlapping_criteria,
            window = self.window,
            similarity_name = self.similarity_name, 
            keep_uppercase = self.keep_uppercase
        )

    def __call__(self, doc: Doc):
        # find all the matches
        
        if self.searcher is None:
            print('Loading SimString Database...')
            self.searcher = SimstringDBReader(self.quick_umls_database, self.similarity_name, self.threshold)
        
        if self.string_to_cui is None:
            print('Loading SimString Database...')
            f = open(self.knowledge_base,'rb')
            self.string_to_cui = pickle.load(f)
            f.close()

        matches = self._match(doc)

        # TODO : 
        # Now update the doc
        # where to put the matches in doc ??
        # make spans  ??
        # maybe in doc.ents ?? or in doc.medical_ents ??
        # definitely put in doc.cuis

        # print(matches)

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
    
        
    def _is_valid_start_token(self, tok):
            return not(
                (self._is_stop_term(tok) and tok.text not in self.negations)
            )

    def _is_stop_term(self, tok):
        return tok.text in self._stopwords

    def _is_valid_end_token(self, tok):
            return not(
                self._is_stop_term(tok)
            )

    def _is_longer_than_min(self, span):
            return (len(span.text.strip())) >= self.min_match_length
    
    def _make_ngrams_detail(self, sent, window : int):

        sent_length = len(sent)

        for i in range(sent_length):
            tok = sent[i]

            if self._is_valid_start_token(tok):
                compensate = False
            else:
                compensate = True

            span_end = min(sent_length, i + window) + 1

            if (
                i + 1 == sent_length and            
                self._is_valid_end_token(tok) and  
                len(tok) >= self.min_match_length
            ):
                yield(i, i+1, tok.text)

            for j in range(i + 1, span_end):
                if compensate:
                    compensate = False
                    continue

                if not self._is_valid_end_token(sent[j - 1]):
                    continue

                span = sent[i:j]

                if not self._is_longer_than_min(span):
                    continue

                yield (
                    i, j,
                    span.text.strip()
                )

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
            matches = list(self.searcher.get(ngram))

            ngram_matches = []
            prev_cui = None
            
            for match in matches:
                
                # get all CUIs with this matched text
                # in database. Yes, one text can have more than
                # one CUI. for eg. 40 year old has CUI of age, adult etc. 
                cuis = self.string_to_cui[match]

                match_similarity = get_similarity(ngram, match, self.ngram_length, self.similarity_name)

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
                all_matches.append(
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
    
    def _make_ngrams(self, doc : Doc, window : int):
        """Make token ngrams of size `window`"""
        for i in range(len(doc)):
            for j in range(i + 1, min(i + self.window, len(doc)) + 1):
                span = doc[i : j]
                yield (span.start, span.end, span.text)

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

        ngrams = list(self._make_ngrams_detail(doc, self.window))

        matches = self._get_all_matches(ngrams)

        if best_match:
            matches = self._select_terms(matches)

        return matches

    @staticmethod
    def simplify(worker: BaseWorker, quick_umls: "QuickUMLS"):
        """Simplifies a QuickUMLS object. 

        Args:
            worker (BaseWorker): The worker on which the
                simplify operation is carried out.
            quick_umls (QuickUMLS): the QuickUMLS object
                to simplify.

        Returns:
            (tuple): The simplified QuickUMLS object.
        """

        quick_umls_database = serde._simplify(worker, quick_umls.quick_umls_database), # Change later
        knowledge_base = serde._simplify(worker, quick_umls.knowledge_base) # Change later
        threshold = serde._simplify(worker, quick_umls.threshold)
        overlapping_criteria = serde._simplify(worker, quick_umls.overlapping_criteria)
        window = serde._simplify(worker, quick_umls.window)
        similarity_name = serde._simplify(worker, quick_umls.similarity_name)
        keep_uppercase = serde._simplify(worker, quick_umls.keep_uppercase)

        return (quick_umls_database, 
                knowledge_base, 
                threshold, 
                overlapping_criteria, 
                window,
                similarity_name,
                keep_uppercase
                )

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Takes a simplified QuickUMLS object, details it 
           and returns a QuickUMLS object.

        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            simple_obj (tuple): the simplified SubPipeline object.
        Returns:
            (QuickUMLS): The QuickUMLS object.
        """

        # Unpack the simplified object
        quick_umls_database, knowledge_base, threshold, overlapping_criteria, window, similarity_name, keep_uppercase = simple_obj

        # Detail each property
        quick_umls_database = serde._detail(worker, quick_umls_database)
        knowledge_base = serde._detail(worker, knowledge_base)
        threshold = serde._detail(worker, threshold)
        overlapping_criteria = serde._detail(worker, overlapping_criteria)
        window = serde._detail(worker, window)
        similarity_name = serde._detail(worker, similarity_name)
        keep_uppercase = serde._detail(worker, keep_uppercase)

        # Instantiate a QuickUMLS object
        quick_umls = QuickUMLS(
            quick_umls_database = self.quick_umls_database, # Change later
            knowledge_base = self.knowledge_base, # Change later
            threshold = self.threshold, 
            overlapping_criteria = self.overlapping_criteria,
            window = self.window,
            similarity_name = self.similarity_name, 
            keep_uppercase = self.keep_uppercase
        )

        return quick_umls