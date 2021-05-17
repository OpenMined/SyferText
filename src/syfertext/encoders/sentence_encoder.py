from ..vocab import Vocab


class SentenceEncoder:
    """This is a simple encoder that takes a text, tokenizes
    it and then uses the vocabulary to convert each token to an
    integer ID.

    IMPORTANT: This encoder does not support excluding tokens.
    """

    def __init__(self, tokenizer, vocab=None):

        self.tokenizer = tokenizer

        # If no vocabulary is specified, then create
        # an empty Vocab object
        if vocab is None:
            self.vocab = Vocab()
        else:
            self.vocab = vocab

    def __call__(self, text):
        """Encode the given text.
        """

        # Intialize the list that will hold the token IDs
        token_ids = []

        # Tokenize
        text_doc = self.tokenizer(text)

        # Convert words to integer ids using
        # the vocabulary
        for token in text_doc:
            token_ids.append(self.vocab.get_id(token.text))

        # Prepare the encoder output
        enc_output = dict(doc=text_doc, token_ids=token_ids)

        return enc_output
