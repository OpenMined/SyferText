from ..tagger import Tagger
from typing import Union

from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde


class EntityRecognizer:
    """This is model-based EntityRecognizer. It tags tokens in given doc/docs with 
    respective entity tags present for the specific model.
    """

    def __init__(self, vocab, model=None):
        """Initialize the EntityRecognizer object.
        Args:
            vocab: `Vocab` of doc
            model: A pytorch model.
        """

        self.vocab = vocab

        # This way of loading method will be updated in future.
        # to install model as package or load from path
        self.model = model

        # doc sequence tagger
        self.tagger = Tagger(vocab=vocab, task="ner", model=model)

    def __call__(self, doc: Doc):

        # doc with tokens entity tags
        doc = self.tagger(doc)

        return doc

    def pipe(self, stream, batch_size=32):
        """Apply the pipe to a stream of documents.
        """

        for docs in utils.batching(stream, bs=batch_size):
            docs = list(docs)
            predictions = self.predict(docs)
            self.set_annotations(docs, predictions)
            yield docs

    def predict(self, docs):
        predictions = self.tagger.predict(docs=docs)

        return predictions

    def set_annotations(self, docs, batch_tags):
        self.tagger.set_annotations(docs, batch_tags)

    def factory(self):
        """Creates a clone of this object.
        This method is used by the SupPipeline class to create
        objects using subpipeline templates.
        """

        return EntityRecognizer(model=self.model)

    # Todo : simplify and deatil functions.
