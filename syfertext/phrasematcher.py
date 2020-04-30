from .doc import Doc
from .vocab import Vocab
from .utils import hash_string
from .tokenizer import Tokenizer

from typing import Union, List

class PhraseMatcher:
    """Efficiently match large terminology lists. The `PhraseMatcher` accepts
    match patterns in the form of `Doc` objects.

    Adapted from FlashText: https://github.com/vi3k6i5/flashtext
    MIT License (see `LICENSE`)
    Copyright (c) 2017 Vikash Singh (vikash.duliajan@gmail.com)
    """

    def __init__(self, vocab : Vocab):
        """Initialize the PhraseMatcher.

        Args:
            vocab (Vocab): The vocabulary object which must be shared with the documents
                matcher will operate on.
        """
        
        # initialize trie for matching
        self.main_trie = {}

        self.vocab = vocab

        # set of hashes of match_id
        self._patterns = set()

        # dictionary of callback functions
        self._callbacks = {}

        # hash used to denote that the pattern has been found
        self._terminal_hash = -1 # TODO : maybe use some other unique hash ?


    def add(self, match_id : str , pattern : str, on_match = None):
        """Add a match-rule to the phrase-matcher. 
        A match-rule consists of: an match ID, and one pattern in the form of doc.

        Args:
            match_id (str): The match ID.
            doc (Doc) : `Doc` object representing match pattern.
        """

        # get the hash of the match_id and store in patterns list
        match_id_hash = self.vocab.store[match_id]
        self._patterns.add(match_id_hash)

        # Add the callback function to self.callbacks
        self._callbacks[match_id_hash] = on_match

        # initialize the tokenizer
        tokenizer = Tokenizer(vocab = self.vocab)

        # tokenize the pattern string
        doc = tokenizer(pattern)

        cur_trie = self.main_trie
        
        for token_meta in doc.container:

            # get the hash of the token
            token_hash = token_meta.orth
            
            if token_hash not in cur_trie:
                cur_trie[token_hash] = {}
            
            cur_trie = cur_trie[token_hash]

        cur_trie[self._terminal_hash] = match_id_hash
    
    
    def __call__(self, doc : Doc):
        """Find all sequences matching the supplied patterns on the `doc` provided. This function
        in turn calls the `find_matches` fucntion where the real work happens. After finding the matches
        it run the callback function of each found match.

        Args:
            doc (Doc): The document to match over.

        Returns: 
            matches (list): A list of `(match_id, start, end)` tuples,
                describing the matches. A match tuple describes a span
                `doc[start:end]`.
        """

        if doc is None or len(doc) == 0:
            # if doc is empty or None just return empty list
            return []

        # find the matches
        matches = self.find_matches(doc)

        # run callback functions on the found matches

        for i,(match_id,start,end) in enumerate(matches):
            # get the callback function
            on_match = self._callbacks[match_id]

            if on_match is not None:
                on_match(self, doc, i, matches)
        
        # return the list of matches found on doc
        return matches

    def find_matches(self, doc: Doc): # TODO : maybe get text from vocab directly instead of creating token first
        """This function finds all the matches over the supplied doc
        
        Args:
            doc (Doc): The document to match over

        Returns: 
            matches (list): A list of `(match_id, start, end)` tuples,
            describing the matches. A match tuple describes a span
            `doc[start:end]`.
        """

        # store the current position of initial trie in cur to store
        # the current position of the trie
        cur = self.main_trie

        # store the start index of the matched pattern
        start = 0

        # the index used to iterate over the doc
        idx = 0

        # helper index used to iterate over the matched pattern
        idy = 0

        # list to store all patterns
        matches = []
        
        while idx < len(doc):

            # consider every index as a potential match, so
            # store the idx in start
            start = idx

            # since we store hash value in the trie structure,
            # we only need the hash of the token at idx position in doc
            token = doc.container[idx].orth
            
            # check if token is in current position in trie
            if token in cur:

                # move down in trie
                cur = cur[token]

                # since token was in trie, this could be a potential match, so we use idy
                # to iterate
                idy = idx + 1
                
                # continue matching as long as possible
                while idy < len(doc):
                    
                    # get hash of token at idy position
                    token = doc.container[idy].orth
                    
                    # if we have terminating hash at this position,
                    # that means we have reached end of one pattern
                    if self._terminal_hash in cur:
                        
                        # get the match_id of the pattern
                        match_id = cur[self._terminal_hash]

                        # append the found pattern span in matches
                        matches.append((match_id,start,idy))
                    
                    # look if the current token is in current trie
                    if token in cur:

                        # if yes, then move down the trie and increase idy by 1
                        cur = cur[token]
                        idy += 1
                    
                    else:
                        # this means token was not in trie, so the 
                        # pattern we were looking for was not found
                        break;
                    
                else:
                    # This is to handle if we reached the end of the doc
                    # If we have a terminal hash at this cur trie, then push that
                    # pattern in matches
                    if self._terminal_hash in cur:
                        # get the match_id of the pattern
                        match_id = cur[self._terminal_hash]

                        # append the found pattern span in matches
                        matches.append((match_id,start,idy))
            
            # reset the trie
            cur = self.main_trie

            # increase idx to move to next token in doc
            idx += 1
        
        # return all found matches
        return matches


    def remove(self,match_id : str):
        """Remove the pattern from the main_trie using the 
        match_id if present.

        Args:
            match_id (str): ID of match to remove

        """
        pass

    def __contains__(self, match_id : str):
        """Checks whether a match pattern corresponding to 
        match_id exists or not

        Args:
            match_id (str) : ID to look for

        Returns:
            Boolean value that ID is present is or not
        """

        if self.vocab.store[match_id] in self._patterns:
            return True
        
        return False

    

