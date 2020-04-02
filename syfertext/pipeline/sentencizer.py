from syfertext.doc import Doc
from syfertext.token import Token

from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde


class Sentencizer:
    """Segment the Doc into sentences using a rule-based strategy.
    """

    def __init__(self, punct_chars=None):
        """Initialize the sentencizer.

        Args:
            punct_chars (list): Punctuation characters to split on. Will be
                                serialized with the nlp object.
        """

        if punct_chars is not None:
            self.punct_chars = punct_chars
        else:
            self.punct_chars = [".", "!", "?"]

    def factory(self):
        """Creates a clone of this object.
        This method is used by the SupPipeline class to create
        objects using subpipeline templates.
        """

        return Sentencizer(punct_chars=self.punct_chars,)

    def __call__(self, doc):
        """Apply the sentencizer to a Doc and set TokenMeta.is_sent_start.

        Args:
            doc (Doc): The document to process.
        
        Returns:
            doc (Doc): The processed Doc.
        """
        # points to the start of sentence
        start = 0

        # bool value to indicate whether we just saw a punctuation
        seen_period = False

        for i, token in enumerate(doc):

            is_in_punct_chars = token.text in self.punct_chars

            # if we just saw a punctuation and current char is not a punctuation
            if seen_period and not is_in_punct_chars:

                # mark doc[start] as start of a sentence
                doc.container[start].is_sent_start = True

                # update start to current index
                start = i

                # set seen punctuation to false
                seen_period = False

            elif is_in_punct_chars:

                # since current char is a punctuation, set seen_period to True
                seen_period = True

        # to handle case if doc doesn't end with a punctuation
        if start < len(doc):
            # set the recent start as start of a sentence
            doc.container[start].is_sent_start = True

        return doc

    @staticmethod
    def simplify(worker: BaseWorker, sentencizer: "Sentencizer"):
        """Simplifies a Sentencizer object. 

        Args:
            worker (BaseWorker): The worker on which the
                simplify operation is carried out.
            sentencizer (Sentencizer): the Sentencizer object
                to simplify.

        Returns:
            (tuple): The simplified Sentencizer object.
        
        """

        # Simplify the object properties
        punct_chars = serde._simplify(worker, sentencizer.punct_chars)

        return (punct_chars,)

    @staticmethod
    def detail(worker: BaseWorker, simple_obj: tuple):
        """Takes a simplified Sentencizer object, details it 
        and returns a Sentencizer object.

        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            simple_obj (tuple): the simplified Sentencizer object.
        Returns:
            (Sentencizer): The Sentencizer object.
        """

        # Unpack the simplified object
        (punct_chars,) = simple_obj

        # Detail each property
        punct_chars = serde._detail(worker, punct_chars)

        # Instantiate a Sentencizer object
        sentencizer = Sentencizer(punct_chars=punct_chars,)

        return sentencizer
