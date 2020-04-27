from . import utils

class Tagger():
    """Class for Sequence tagging.
    currently only supports for SyferText ner model.

    ToDo :1.Extend to pos and custom tagging with any model.
        2.Make further training of model possible.

    """

    def __init__(self, vocab, model, task='ner'):
        self.vocab = vocab
        self.task = task
        self.model = model

    def __call__(self,doc):
        tags = self.predict([doc])
        self.set_annotations([doc], tags)
        return doc

    def tok2vec(self, docs):
        "Vectorize stream of doc to pass it in model"
        #Todo
        pass

    def predict(self,docs):
      
        tokvecs = self.tok2vec(docs)
        scores = self.model(tokvecs)
        predictions = []
        for doc_scores in scores:
            doc predictions = doc_scores.argmax(axis=1)
            predictions.append(doc predictions)
        return  predictions

    def set_attribute(self, obj:object, name: str, value: object):
        """Creates a custom attribute with the name `name` and
           value `value` in the Underscore object `self._`

        Args:
            name (str): name of the custom attribute.
            value (object): value of the custom named attribute.
        """

        # make sure there is no space in name as well prevent empty name
        assert (
            isinstance(name, str) and len(name) > 0 and (not (" " in name))
        ), "Argument name should be a non-empty str type containing no spaces"

        setattr(obj, name, value)

    def set_annotations(self, docs, batch_tags):

        if self.task == "ner":
            attr_name = "ent"
        else attr_name = self.task

        for i, doc in enumerate(docs):
            doc_tags = batch_tags[i]

            for j, tag_id in enumerate(doc_tags):
                tag = self.vocab.strings[self.entity_labels[tag_id]]
                self.set_attribute(obj=doc[j], name=attr_name, value = tag)

    def pipe(self, stream, batch_size=32):
        """Apply the pipe to a stream of documents.
        """

        for docs in utils.batching(stream, bs=batch_size):
            docs = list(docs)
            predictions = self.predict(docs)
            self.set_annotations(docs, predictions)
            yield docs

    