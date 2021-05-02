class Vocab:
    """A class that represents a vocabulary
    """

    def __init__(self):

        self.text2id = dict()
        self.id2text = dict()

    def add(self, text: str):
        """Adds a word to the vocab
        """

        # Compute the ID of the added word
        current_id = len(self.text2id)

        # Update the dicts only if the word is not yet known
        self.text2id[text] = self.text2id.get(text, current_id)
        self.id2text[current_id] = self.id2text.get(current_id, text)

    def get_id(self, text):

        if text not in self.text2id:
            self.add(text)

        return self.text2id[text]
